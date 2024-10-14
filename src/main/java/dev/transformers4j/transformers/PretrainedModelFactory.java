package dev.transformers4j.transformers;

import com.google.gson.GsonBuilder;
import com.google.gson.reflect.TypeToken;
import dev.transformers4j.transformers.utils.GsonObjectToNumberStrategy;
import io.vavr.Tuple;
import io.vavr.Tuple2;
import io.vavr.control.Either;
import org.apache.commons.lang3.ObjectUtils;
import org.apache.commons.lang3.reflect.FieldUtils;
import org.semver4j.Semver;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.lang.reflect.InvocationTargetException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;
import java.util.regex.Pattern;

import static dev.transformers4j.Init.__version__;
import static dev.transformers4j.transformers.utils.Generic.add_model_info_to_auto_map;
import static dev.transformers4j.transformers.utils.Hub.cached_file;
import static dev.transformers4j.transformers.utils.Hub.download_url;
import static dev.transformers4j.transformers.utils.Hub.is_remote_url;
import static dev.transformers4j.transformers.utils.Init.CONFIG_NAME;

public class PretrainedModelFactory<T extends PretrainedModel> {
    private static final Logger LOGGER = LoggerFactory.getLogger(PretrainedModelFactory.class);

    public T create(Map<String, Object> kwargs) {
        throw new UnsupportedOperationException("Not implemented");
    }
}
