package dev.transformers4j.transformers.models.auto;

import dev.transformers4j.transformers.models.bert.BertConfig;
import dev.transformers4j.transformers.models.roberta.RobertaConfig;
import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.nio.file.Path;
import java.util.Collections;
import java.util.HashMap;

import static dev.transformers4j.transformers.TestingUtils.DUMMY_UNKNOWN_IDENTIFIER;
import static dev.transformers4j.transformers.TestingUtils.get_tests_dir;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;

public class AutoConfigTest {
    private static final Path SAMPLE_ROBERTA_CONFIG = get_tests_dir("fixtures/dummy-config.json");

    @Test
    public void test_config_from_model_shortcut() throws IOException {
        var config = AutoConfig.from_pretrained(Path.of("google-bert/bert-base-uncased"));
        assertInstanceOf(BertConfig.class, config);
    }

    @Test
    public void test_config_model_type_from_local_file() throws IOException {
        var config = AutoConfig.from_pretrained(SAMPLE_ROBERTA_CONFIG);
        assertInstanceOf(RobertaConfig.class, config);
    }

    @Test
    public void test_config_model_type_from_model_identifier() throws IOException {
        var config = AutoConfig.from_pretrained(DUMMY_UNKNOWN_IDENTIFIER);
        assertInstanceOf(RobertaConfig.class, config);
    }

    @Test
    public void test_config_for_model_str() throws IOException {
        var config = AutoConfig.for_model("roberta", new HashMap<>());
        assertInstanceOf(RobertaConfig.class, config);
    }
}
