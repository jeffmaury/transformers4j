package dev.transformers4j.transformers.models.roberta;

import dev.transformers4j.transformers.PretrainedConfigFactory;
import dev.transformers4j.transformers.models.auto.ModelType;

import java.util.Map;

@ModelType("roberta")
public class RobertaConfigFactory extends PretrainedConfigFactory<RobertaConfig> {
    @Override
    public RobertaConfig create(Map<String, Object> kwargs) {
        return new RobertaConfig(kwargs);
    }
}
