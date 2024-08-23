package dev.transformers4j.transformers;

import java.net.URISyntaxException;
import java.nio.file.Path;
import java.nio.file.Paths;

public class TestingUtils {
    public static Path get_tests_dir(String append_path) {
        try {
            var resource = append_path != null ? append_path : "/";
            return Paths.get(TestingUtils.class.getClassLoader().getResource(resource).toURI());
        } catch (URISyntaxException e) {
            throw new RuntimeException(e);
        }
    }
}
