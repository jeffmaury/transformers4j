package dev.transformers4j.transformers;

import java.util.Map;

public class DefaultPretrainedConfigFactory<C extends PretrainedConfig> extends PretrainedConfigFactory<C> {
    private final Class<? extends PretrainedConfig> configClass;

    public DefaultPretrainedConfigFactory(Class<C> configClass) {
        this.configClass = configClass;
    }

    public C createConfig(Map<String, Object> kwargs) {
        try {
            return (C) configClass.getConstructor(Map.class).newInstance(kwargs);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }
}
