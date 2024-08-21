package dev.transformers4j.transformers.models.auto;

import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.nio.file.Path;
import java.util.Collections;

public class AutoConfigTest {
    @Test
    public void test_config_from_model_shortcut() throws IOException {
        var config = AutoConfig.from_pretrained(Path.of("google-bert/bert-base-uncased"));
    }

}
