package dev.transformers4j.transformers.models.bert;

import dev.transformers4j.transformers.PretrainedConfigFactory;
import dev.transformers4j.transformers.models.auto.ModelType;

import java.util.Map;

@ModelType("bert")
public class BertConfigFactory extends PretrainedConfigFactory<BertConfig> {
    @Override
    public BertConfig create(Map<String, Object> kwargs) {
        return new BertConfig(kwargs);
    }
}
