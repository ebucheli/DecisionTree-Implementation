import numpy as np

if __package__:
    from .classifier import DecisionTreeClassifier
    from .node import Node
else:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from decision_tree.classifier import DecisionTreeClassifier
    from decision_tree.node import Node

data = {
    "stream": ["false", "true", "true", "false", "false", "true", "true"],
    "slope": ["steep", "moderate", "steep", "steep", "flat", "steep", "steep"],
    "elevation": ["high", "low", "medium", "medium", "high", "highest", "high"],
    "vegetation": [
        "chapparal", "riparian", "riparian", "chapparal",
        "conifer", "conifer", "chapparal",
    ],
}

feature_names = ["stream", "slope", "elevation"]

# código numérico como en el primer arbol de decision
encoders = {
    "stream": {"false": 0, "true": 1},
    "slope": {"flat": 0, "moderate": 1, "steep": 2},
    "elevation": {"high": 0, "highest": 1, "low": 2, "medium": 3},
}

X = np.array([
    [encoders[name][data[name][row]] for name in feature_names]
    for row in range(len(data["vegetation"]))
])
y = np.array(data["vegetation"])

tree = DecisionTreeClassifier(max_depth=2)
tree.root = Node(
    feature_index=0,  # split 1, depth 0: stream false vs. true
    threshold=0.5,
    left=Node(
        feature_index=1,  # split 2, depth 1: flat vs. moderate/steep
        threshold=0.5,
        left=Node(value="conifer"),
        right=Node(value="chapparal"),
    ),
    right=Node(value="riparian"),
)

print("Category mappings:", encoders)
print("Training predictions:", tree.predict(X))
print("Training accuracy:", tree.eval(X, y))

sample = {"stream": "true", "slope": "moderate", "elevation": "low"} # para hacer la selección de la muestrq manual
encoded_sample = [[encoders[name][sample[name]] for name in feature_names]]
print("Prediction for", sample, ":", tree.predict(encoded_sample)[0])