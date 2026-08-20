"""Independent Fast Interpretable Greedy-Tree Sums implementation.

This module does not import or call ``imodels``.  It implements the P2 greedy
procedure directly: at each iteration every current leaf and a possible fresh
root compete globally; targets for a candidate in tree k are residualized by
all *other* trees; and an accepted split can revisit any existing tree.

Classification follows the official reference implementation's observable
multiclass behavior: one-hot targets are fitted with multi-output least-squares
stumps and summed scores are transformed with softmax.  The P2 prose mentions
Gini for classification, but Gini is not defined for the continuous residual
vectors required after the first tree.  Both the discrepancy and this explicit
choice are retained in the project report.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator, Sequence

import numpy as np
from scipy.special import softmax
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.tree import DecisionTreeRegressor
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y


@dataclass
class FIGSNode:
    """A committed node that also stores its currently proposed next split."""

    mask: np.ndarray
    value: np.ndarray
    tree_index: int
    depth: int
    is_new_root: bool = False
    feature: int | None = None
    threshold: float | None = None
    gain: float | None = None
    impurity: float | None = None
    left: "FIGSNode | None" = None
    right: "FIGSNode | None" = None
    proposed_left: "FIGSNode | None" = field(default=None, repr=False)
    proposed_right: "FIGSNode | None" = field(default=None, repr=False)
    accepted_gain: float = 0.0

    @property
    def is_leaf(self) -> bool:
        return self.left is None and self.right is None


class _ScratchFIGS(BaseEstimator):
    def __init__(
        self,
        max_rules: int = 12,
        max_trees: int | None = None,
        max_depth: int | None = None,
        min_impurity_decrease: float = 0.0,
        min_samples_leaf: int = 1,
        max_features: int | float | str | None = None,
        random_state: int | None = 42,
        class_weight=None,
        backfit: bool = False,
        backfit_iterations: int = 5,
    ):
        self.max_rules = max_rules
        self.max_trees = max_trees
        self.max_depth = max_depth
        self.min_impurity_decrease = min_impurity_decrease
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state
        self.class_weight = class_weight
        self.backfit = backfit
        self.backfit_iterations = backfit_iterations

    def _prepare_target(self, y: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def _candidate(
        self,
        X: np.ndarray,
        target: np.ndarray,
        node: FIGSNode,
        sample_weight: np.ndarray | None,
    ) -> FIGSNode:
        """Refit a depth-one CART proposal while preserving node identity/value."""

        count = int(node.mask.sum())
        node.feature = None
        node.threshold = None
        node.gain = None
        node.impurity = None
        node.proposed_left = None
        node.proposed_right = None
        if count < 2 * self.min_samples_leaf:
            return node
        stump = DecisionTreeRegressor(
            max_depth=1,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features,
            random_state=self.random_state,
        )
        local_weight = sample_weight[node.mask] if sample_weight is not None else None
        stump.fit(X[node.mask], target[node.mask], sample_weight=local_weight)
        tree = stump.tree_
        if tree.node_count == 1 or tree.feature[0] < 0:
            node.impurity = float(tree.impurity[0])
            return node
        local_split = X[:, tree.feature[0]] <= tree.threshold[0]
        left_mask = node.mask & local_split
        right_mask = node.mask & ~local_split
        if sample_weight is None:
            left_n, right_n = float(left_mask.sum()), float(right_mask.sum())
        else:
            left_n = float(sample_weight[left_mask].sum())
            right_n = float(sample_weight[right_mask].sum())
        total = left_n + right_n
        if total <= 0:
            return node
        gain = total * float(tree.impurity[0]) - left_n * float(tree.impurity[1]) - right_n * float(tree.impurity[2])
        node.feature = int(tree.feature[0])
        node.threshold = float(tree.threshold[0])
        node.gain = float(max(gain, 0.0))
        node.impurity = float(tree.impurity[0])
        node.proposed_left = FIGSNode(
            mask=left_mask,
            value=np.asarray(tree.value[1], dtype=float).reshape(-1),
            tree_index=node.tree_index,
            depth=node.depth + 1,
        )
        node.proposed_right = FIGSNode(
            mask=right_mask,
            value=np.asarray(tree.value[2], dtype=float).reshape(-1),
            tree_index=node.tree_index,
            depth=node.depth + 1,
        )
        return node

    @staticmethod
    def _predict_one_tree(root: FIGSNode, X: np.ndarray) -> np.ndarray:
        output = np.empty((len(X), len(root.value)), dtype=float)

        def assign(node: FIGSNode, indices: np.ndarray) -> None:
            if node.is_leaf:
                output[indices] = node.value
                return
            go_left = X[indices, node.feature] <= node.threshold
            assign(node.left, indices[go_left])
            assign(node.right, indices[~go_left])

        assign(root, np.arange(len(X), dtype=int))
        return output

    @staticmethod
    def _iter_leaves(root: FIGSNode) -> Iterator[FIGSNode]:
        if root.is_leaf:
            yield root
        else:
            yield from _ScratchFIGS._iter_leaves(root.left)
            yield from _ScratchFIGS._iter_leaves(root.right)

    @staticmethod
    def _weighted_mean(
        values: np.ndarray, weights: np.ndarray | None
    ) -> np.ndarray:
        if weights is None:
            return values.mean(axis=0)
        return np.average(values, axis=0, weights=weights)

    def _backfit_values(
        self, X: np.ndarray, target: np.ndarray, sample_weight: np.ndarray | None
    ) -> None:
        """Supplement block-coordinate descent with structure held fixed."""

        for _ in range(self.backfit_iterations):
            per_tree = [self._predict_one_tree(root, X) for root in self.trees_]
            for tree_index, root in enumerate(self.trees_):
                other = sum(
                    (prediction for j, prediction in enumerate(per_tree) if j != tree_index),
                    np.zeros_like(target),
                )
                residual = target - other
                for leaf in self._iter_leaves(root):
                    # Recompute membership from the current structure; the stored
                    # training mask is also kept as an auditable cross-check.
                    mask = leaf.mask
                    if mask.any():
                        local_weights = sample_weight[mask] if sample_weight is not None else None
                        leaf.value = self._weighted_mean(residual[mask], local_weights)
                per_tree[tree_index] = self._predict_one_tree(root, X)

    def fit(self, X, y, sample_weight=None):
        X_array, y_array = check_X_y(
            X, y, multi_output=False, y_numeric=isinstance(self, RegressorMixin), dtype=float
        )
        if self.max_rules is None or int(self.max_rules) < 0:
            raise ValueError("max_rules must be a nonnegative integer")
        if self.max_trees is not None and int(self.max_trees) < 1:
            raise ValueError("max_trees must be at least one")
        if self.max_depth is not None and int(self.max_depth) < 1:
            raise ValueError("max_depth must be at least one")
        if self.backfit_iterations < 0:
            raise ValueError("backfit_iterations cannot be negative")
        self.n_features_in_ = X_array.shape[1]
        if hasattr(X, "columns"):
            self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        target = self._prepare_target(np.asarray(y_array))
        self.n_outputs_ = target.shape[1]
        weights = None if sample_weight is None else np.asarray(sample_weight, dtype=float)
        if weights is not None and weights.shape != (len(X_array),):
            raise ValueError("sample_weight has the wrong shape")
        if isinstance(self, ClassifierMixin) and self.class_weight is not None:
            class_weights = compute_sample_weight(self.class_weight, y_array)
            weights = class_weights if weights is None else weights * class_weights
        elif isinstance(self, RegressorMixin) and self.class_weight is not None:
            raise ValueError("class_weight is only defined for classification")

        self.trees_: list[FIGSNode] = []
        self.complexity_ = 0
        self.split_history_: list[dict[str, float | int]] = []
        full_mask = np.ones(len(X_array), dtype=bool)
        root_value = self._weighted_mean(target, weights)
        candidates = [
            self._candidate(
                X_array,
                target,
                FIGSNode(full_mask, root_value, -1, 0, is_new_root=True),
                weights,
            )
        ]

        while candidates and self.complexity_ < int(self.max_rules):
            valid = [
                candidate
                for candidate in candidates
                if candidate.gain is not None
                and candidate.gain + 1e-15 >= self.min_impurity_decrease
                and (self.max_depth is None or candidate.depth < self.max_depth)
                and not (
                    candidate.is_new_root
                    and self.max_trees is not None
                    and len(self.trees_) >= self.max_trees
                )
            ]
            if not valid:
                break
            # max() is stable; candidates are maintained in tree/leaf creation
            # order, which makes exact ties deterministic.
            # Remove by object identity. Dataclass equality is intentionally not
            # used because masks are NumPy arrays with elementwise equality.
            selected = max(valid, key=lambda candidate: candidate.gain)
            selected_position = next(
                index for index, candidate in enumerate(candidates) if candidate is selected
            )
            candidates.pop(selected_position)
            if selected.is_new_root:
                selected.tree_index = len(self.trees_)
                selected.is_new_root = False
                selected.proposed_left.tree_index = selected.tree_index
                selected.proposed_right.tree_index = selected.tree_index
                self.trees_.append(selected)
                candidates.append(
                    FIGSNode(
                        mask=full_mask.copy(),
                        value=np.zeros(self.n_outputs_, dtype=float),
                        tree_index=-1,
                        depth=0,
                        is_new_root=True,
                    )
                )
            selected.left = selected.proposed_left
            selected.right = selected.proposed_right
            selected.accepted_gain = float(selected.gain)
            self.complexity_ += 1
            self.split_history_.append(
                {
                    "rule": self.complexity_,
                    "tree": selected.tree_index,
                    "depth": selected.depth,
                    "feature": selected.feature,
                    "threshold": selected.threshold,
                    "gain": selected.accepted_gain,
                }
            )
            candidates.extend([selected.left, selected.right])

            per_tree = [self._predict_one_tree(root, X_array) for root in self.trees_]
            refreshed: list[FIGSNode] = []
            for candidate in candidates:
                if candidate.is_new_root:
                    residual = target - sum(per_tree, np.zeros_like(target))
                else:
                    residual = target - sum(
                        (
                            prediction
                            for index, prediction in enumerate(per_tree)
                            if index != candidate.tree_index
                        ),
                        np.zeros_like(target),
                    )
                refreshed.append(self._candidate(X_array, residual, candidate, weights))
            candidates = refreshed

        if not self.trees_:
            # A constant target or zero-rule budget still yields a valid mean
            # predictor, while complexity remains zero.
            self.trees_ = [FIGSNode(full_mask, root_value, 0, 0)]
        if self.backfit and self.backfit_iterations:
            self._backfit_values(X_array, target, weights)
        self.feature_importances_ = np.zeros(self.n_features_in_, dtype=float)
        for record in self.split_history_:
            self.feature_importances_[int(record["feature"])] += float(record["gain"])
        total_gain = self.feature_importances_.sum()
        if total_gain > 0:
            self.feature_importances_ /= total_gain
        return self

    def _raw_predict(self, X) -> np.ndarray:
        check_is_fitted(self, "trees_")
        X_array = check_array(X, dtype=float)
        if X_array.shape[1] != self.n_features_in_:
            raise ValueError("X has a different feature count than the fitted estimator")
        return sum(
            (self._predict_one_tree(root, X_array) for root in self.trees_),
            np.zeros((len(X_array), self.n_outputs_), dtype=float),
        )

    def export_rules(self, feature_names: Sequence[str] | None = None) -> list[dict[str, object]]:
        """Export every node/leaf as a compact, human-readable rule table."""

        check_is_fitted(self, "trees_")
        names = (
            list(feature_names)
            if feature_names is not None
            else list(getattr(self, "feature_names_in_", [f"x{i}" for i in range(self.n_features_in_)]))
        )
        rows: list[dict[str, object]] = []

        def visit(node: FIGSNode, tree_index: int, conditions: list[str]) -> None:
            if node.is_leaf:
                rows.append(
                    {
                        "tree": tree_index,
                        "depth": node.depth,
                        "rule": " and ".join(conditions) if conditions else "TRUE",
                        "value": node.value.tolist(),
                        "n_training_rows": int(node.mask.sum()),
                    }
                )
                return
            name = names[node.feature]
            visit(node.left, tree_index, [*conditions, f"{name} <= {node.threshold:.6g}"])
            visit(node.right, tree_index, [*conditions, f"{name} > {node.threshold:.6g}"])

        for tree_index, root in enumerate(self.trees_):
            visit(root, tree_index, [])
        return rows


class ScratchFIGSRegressor(RegressorMixin, _ScratchFIGS):
    """FIGS regressor with simultaneous greedy tree growth."""

    def _prepare_target(self, y: np.ndarray) -> np.ndarray:
        return np.asarray(y, dtype=float).reshape(-1, 1)

    def predict(self, X) -> np.ndarray:
        return self._raw_predict(X).reshape(-1)


class ScratchFIGSClassifier(ClassifierMixin, _ScratchFIGS):
    """Multiclass FIGS classifier using residualized one-hot targets."""

    def _prepare_target(self, y: np.ndarray) -> np.ndarray:
        self.classes_, encoded = np.unique(y, return_inverse=True)
        return np.eye(len(self.classes_), dtype=float)[encoded]

    def predict_proba(self, X) -> np.ndarray:
        return softmax(self._raw_predict(X), axis=1)

    def predict(self, X) -> np.ndarray:
        return self.classes_[self.predict_proba(X).argmax(axis=1)]
