package dev.transformers4j.hub.utils;

import java.net.http.HttpResponse;

public class BadRequestException extends HfHubHTTPException {
    public BadRequestException(String message, HttpResponse<?> response) {
        super(message, response);
    }
}
