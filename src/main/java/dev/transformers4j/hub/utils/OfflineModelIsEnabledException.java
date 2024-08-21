package dev.transformers4j.hub.utils;

import java.net.ConnectException;

public class OfflineModelIsEnabledException extends ConnectException {
    public OfflineModelIsEnabledException(String message) {
        super(message);
    }
}
