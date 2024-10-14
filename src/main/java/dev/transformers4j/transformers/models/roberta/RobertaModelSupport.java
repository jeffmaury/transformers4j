package dev.transformers4j.transformers.models.roberta;

import dev.transformers4j.transformers.DefaultPretrainedConfigFactory;
import dev.transformers4j.transformers.PretrainedConfig;
import dev.transformers4j.transformers.PretrainedConfigFactory;
import dev.transformers4j.transformers.models.auto.ModelSupport;
import dev.transformers4j.transformers.models.auto.ModelType;

@ModelType("roberta")
public class RobertaModelSupport implements ModelSupport {
    @Override
    public <T extends PretrainedConfigFactory<? extends PretrainedConfig>> T getConfigFactory() {
        return (T) new DefaultPretrainedConfigFactory(RobertaConfig.class);
    }
}
