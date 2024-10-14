package dev.transformers4j.transformers.models.auto;

import dev.transformers4j.transformers.PretrainedConfig;
import dev.transformers4j.transformers.PretrainedConfigFactory;

import java.util.ServiceLoader;

public interface ModelSupport {
    static ModelSupport getModelSupport(String modelType) {
        var loader = ServiceLoader.load(ModelSupport.class);
        return loader.stream().filter(config -> {
            var modelTypeAnnotation = config.type().getAnnotation(ModelType.class);
            if (modelTypeAnnotation != null) {
                for (String type : modelTypeAnnotation.value()) {
                    if (type.equals(modelType)) {
                        return true;
                    }
                }
            }
            return false;
        }).findFirst().map(ServiceLoader.Provider::get).orElse(null);

    }

    default <T extends PretrainedConfigFactory<? extends PretrainedConfig>> T getConfigFactory() {
        throw new UnsupportedOperationException("Not implemented");
    }

    default <T extends PretrainedModelFactory<? extends PretrainedModel>> T getModelFactory() {
        throw new UnsupportedOperationException("Not implemented");
    }
}
