package dev.transformers4j.transformers;

import org.apache.commons.lang3.ObjectUtils;

import java.util.Map;

public class MapUtil {
    public static <T> T get(Map<String, Object> kwargs, String key, Class<T> type, T def, boolean remove) {
        if (kwargs.containsKey(key)) {
            Object value = kwargs.get(key);
            if (value != null) {
                if (type.isInstance(value)) {
                    return remove ? (T) kwargs.remove(key) : (T) kwargs.get(key);
                } else {
                    throw new IllegalArgumentException(
                            "Expected value of type " + type + " for key " + key + " but got " + value.getClass());
                }
            }
        }
        return def;
    }

    public static <T> T get(Map<String, Object> kwargs, String key, Class<T> type, T def) {
        return get(kwargs, key, type, def, false);
    }

    public static <T> T pop(Map<String, Object> kwargs, String key, Class<T> type, T def) {
        return get(kwargs, key, type, def, true);
    }

    public static Map<String, Object> merge(Map<String, Object> original, String name, Object value) {
        var copy = ObjectUtils.clone(original);
        copy.put(name, value);
        return copy;
    }

    public static Map<String, Object> merge(Map<String, Object> first, Map<String, Object> second) {
        var copy = ObjectUtils.clone(first);
        copy.putAll(second);
        return copy;
    }
}
