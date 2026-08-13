import ast


def load_yaml_config(path):
    try:
        import yaml

        with open(path, "r") as f:
            return yaml.safe_load(f) or {}
    except ModuleNotFoundError:
        return _load_simple_yaml(path)


def _load_simple_yaml(path):
    config = {}
    stack = [(-1, config)]

    with open(path, "r") as f:
        for line_number, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue

            indent = len(line) - len(line.lstrip(" "))
            key, separator, value = stripped.partition(":")
            if separator == "":
                raise ValueError(f"Invalid config line {line_number}: {line.rstrip()}")

            while indent <= stack[-1][0]:
                stack.pop()

            parent = stack[-1][1]
            key = key.strip()
            value = value.split("#", 1)[0].strip()

            if value == "":
                child = {}
                parent[key] = child
                stack.append((indent, child))
            else:
                parent[key] = _parse_scalar(value)

    return config


def _parse_scalar(value):
    lowered = value.lower()
    if lowered in ("true", "false"):
        return lowered == "true"
    if lowered in ("none", "null"):
        return None

    try:
        return ast.literal_eval(value)
    except (SyntaxError, ValueError):
        return value.strip("'\"")
