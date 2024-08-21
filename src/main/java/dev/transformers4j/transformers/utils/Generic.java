package dev.transformers4j.transformers.utils;

import io.vavr.Tuple;

import java.util.Collection;
import java.util.Map;

public class Generic {
    /**
     * Adds the information of the repo_id to a given auto map.
     */
    public static Map<String, Object> add_model_info_to_auto_map(Map<String, Object> auto_map, String repo_id) {
        for (var entry : auto_map.entrySet()) {
            var key = entry.getKey();
            var value = entry.getValue();
            if (value instanceof Tuple) {
                auto_map.put(key, ((Tuple) value).toSeq().asJava().stream()
                        .map(v -> v != null && !v.toString().contains("--") ? repo_id + "--" + v : v));
            } else if (value instanceof Collection) {
                auto_map.put(key, ((Collection) value).stream()
                        .map(v -> v != null && !v.toString().contains("--") ? repo_id + "--" + v : v));
            } else if (value != null && !value.toString().contains("--")) {
                auto_map.put(key, repo_id + "--" + "value");
            }
        }
        return auto_map;
    }
}
