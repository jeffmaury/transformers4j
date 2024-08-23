package dev.transformers4j.transformers.utils;

import com.google.gson.ToNumberStrategy;
import com.google.gson.stream.JsonReader;

import java.io.IOException;

public class GsonObjectToNumberStrategy implements ToNumberStrategy {
    @Override
    public Number readNumber(JsonReader in) throws IOException {
        var value = in.nextString();
        try {
            return Integer.parseInt(value);
        } catch (NumberFormatException e) {
            try {
                return Long.parseLong(value);
            } catch (NumberFormatException e1) {
                return Double.parseDouble(value);
            }
        }
    }
}
