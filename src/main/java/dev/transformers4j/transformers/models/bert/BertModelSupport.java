package dev.transformers4j.transformers.models.bert;

import dev.transformers4j.transformers.DefaultPretrainedConfigFactory;
import dev.transformers4j.transformers.PretrainedConfig;
import dev.transformers4j.transformers.PretrainedConfigFactory;
import dev.transformers4j.transformers.models.auto.ModelSupport;
import dev.transformers4j.transformers.models.auto.ModelType;

import java.util.Map;

@ModelType("bert")
public class BertModelSupport implements ModelSupport {
    @Override
    public <T extends PretrainedConfigFactory<? extends PretrainedConfig>> T getConfigFactory() {
        return (T) new DefaultPretrainedConfigFactory(BertConfig.class);
    }
}
