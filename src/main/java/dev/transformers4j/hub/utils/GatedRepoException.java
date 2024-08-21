package dev.transformers4j.hub.utils;

import java.net.http.HttpResponse;

public class GatedRepoException extends RepositoryNotFoundException {
    public GatedRepoException(String message, HttpResponse<?> response) {
        super(message, response);
    }
}
