from dataclasses import dataclass
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class DecisionConfig:
    """Configuration governing churn classification and retention decisions."""
    # Business values
    customer_value: float
    cost_discount: float
    cost_contact: float

    # Effectiveness
    discount_uplift: float
    contact_uplift: float
    max_discount_budget: float
    max_discount_rate: float = 0.2

    @classmethod
    def from_config(cls, cfg: dict):
        """Build decision settings from the effective project configuration."""
        decision_cfg = cfg.get("decision", {})

        return cls(
            customer_value=float(decision_cfg.get("customer_value", 100)),
            cost_discount=float(decision_cfg.get("cost_discount", 10)),
            cost_contact=float(decision_cfg.get("cost_contact", 2.0)),
            discount_uplift=float(decision_cfg.get("discount_uplift", 0.3)),
            contact_uplift=float(decision_cfg.get("contact_uplift", 0.1)),
            max_discount_rate=float(decision_cfg.get("max_discount_rate", 0.2)),
            max_discount_budget=float(decision_cfg.get("max_discount_budget", 10)),
        )


class DecisionEngine:
    """
    Convert churn probabilities into economically ranked retention actions.

    The engine compares the expected value of available actions using customer
    value, intervention cost, treatment effectiveness and decision thresholds.
    """
    def __init__(self, config: DecisionConfig):
        self.config = config

    def decide(self, p: float, customer_value: float | None = None) -> dict:
        """
        Choose best action based on expected value.
        Supports customer-specific value.
        """

        value = self._resolve_customer_value(customer_value)

        actions = {
            "offer_discount": self._value_discount(p, value),
            "send_email": self._value_contact(p, value),
            "no_action": self._value_no_action(p),
        }

        best_action = max(actions, key=actions.get)
        best_value = actions[best_action]

        if best_value <= 0:
            best_action = "no_action"
            best_value = 0.0

        return {
            "churn_probability": float(p),
            "customer_value": float(value),
            "action": best_action,
            "expected_value": float(best_value),
            "all_actions": actions,
        }

    # ---- VALUE FUNCTIONS ----

    def _resolve_customer_value(self, customer_value: float | None = None) -> float:
        if customer_value is None:
            return float(self.config.customer_value)

        return float(customer_value)

    def _value_discount(self, p: float, customer_value: float) -> float:
        return (
            p * (self.config.discount_uplift * customer_value)
            - self.config.cost_discount
        )

    def _value_contact(self, p: float, customer_value: float) -> float:
        return (
            p * (self.config.contact_uplift * customer_value)
            - self.config.cost_contact
        )

    def _value_no_action(self, p: float) -> float:
        return 0.0

    def decide_batch(
        self,
        probs: list[float],
        customer_values: list[float] | None = None,
    ) -> list[dict]:
        """
        Budget-aware decision using incremental value.
        Supports customer-specific values.
        """

        if customer_values is None:
            customer_values = [self.config.customer_value] * len(probs)

        if len(customer_values) != len(probs):
            raise ValueError("customer_values must have same length as probs")

        candidates = []

        # 1. Compute values per customer
        for i, p in enumerate(probs):
            customer_value = self._resolve_customer_value(customer_values[i])

            v_discount = self._value_discount(p, customer_value)
            v_email = self._value_contact(p, customer_value)
            v_none = self._value_no_action(p)

            fallback_value = max(v_email, v_none)
            fallback_action = "send_email" if v_email >= v_none else "no_action"

            uplift = v_discount - fallback_value

            candidates.append({
                "idx": i,
                "p": float(p),
                "customer_value": float(customer_value),
                "v_discount": float(v_discount),
                "v_email": float(v_email),
                "v_none": float(v_none),
                "fallback_action": fallback_action,
                "fallback_value": float(fallback_value),
                "uplift": float(uplift),
            })

        # 2. Sort by uplift, not probability
        candidates_sorted = sorted(
            candidates,
            key=lambda x: x["uplift"],
            reverse=True,
        )

        # 3. Budget constraint
        selected = set()
        used_budget = 0.0

        if self.config.max_discount_budget:
            budget = self.config.max_discount_budget
            cost_per_discount = self.config.cost_discount

            for c in candidates_sorted:
                if c["uplift"] <= 0:
                    continue

                if used_budget + cost_per_discount > budget:
                    break

                selected.add(c["idx"])
                used_budget += cost_per_discount

        elif self.config.max_discount_rate:
            max_discount = int(len(probs) * self.config.max_discount_rate)
            selected = {
                c["idx"]
                for c in candidates_sorted[:max_discount]
                if c["uplift"] > 0
            }
            used_budget = len(selected) * self.config.cost_discount

        # 4. Final decisions
        results = []

        for c in candidates:
            if c["idx"] in selected:
                action = "offer_discount"
                value = c["v_discount"]
            else:
                action = c["fallback_action"]
                value = c["fallback_value"]

            results.append({
                "churn_probability": c["p"],
                "customer_value": c["customer_value"],
                "action": action,
                "expected_value": float(value),
                "all_actions": {
                    "offer_discount": c["v_discount"],
                    "send_email": c["v_email"],
                    "no_action": c["v_none"],
                },
            })

        logger.info(
            f"Selected {len(selected)} discounts | budget_used={used_budget:.2f}"
        )

        return results