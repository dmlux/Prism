from prism.schema.token_tasks import TokenTaskSchema


TOKEN_TASK_SCHEMA_FORMAT_VERSION = 1


def serialize_token_task_schema(schema: TokenTaskSchema) -> dict[str, object]:
    return {
        "format_version": TOKEN_TASK_SCHEMA_FORMAT_VERSION,
        "upos": {
            "version": schema.upos.version,
            "labels": list(schema.upos.labels),
        },
        "morphology": {
            "version": schema.morphology.version,
            "features": [
                {
                    "name": feature.name,
                    "values": list(feature.values),
                    "allows_multiple_values": (feature.allows_multiple_values),
                }
                for feature in schema.morphology.features
            ],
        },
        "lemma_rules": {
            "version": schema.lemma_rules.version,
            "rules": [
                {
                    "prefix_removal": rule.prefix_removal,
                    "suffix_removal": rule.suffix_removal,
                    "prefix_addition": rule.prefix_addition,
                    "suffix_addition": rule.suffix_addition,
                }
                for rule in schema.lemma_rules.rules
            ],
        },
    }
