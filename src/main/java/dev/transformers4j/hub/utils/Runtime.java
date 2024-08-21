package dev.transformers4j.hub.utils;

import java.util.HashMap;
import java.util.Map;

import static java.util.Map.entry;

public class Runtime {
    private static Map<String, String> _package_versions = new HashMap<>();

    private static Map<String, String[]> _CANDIDATES = Map.ofEntries(entry("aiohttp", new String[] { "aiohttp" }),
            entry("fastai", new String[] { "fastai" }), entry("fastapi", new String[] { "fastapi" }),
            entry("fastcore", new String[] { "fastcore" }), entry("gradio", new String[] { "gradio" }),
            entry("graphviz", new String[] { "graphviz" }), entry("hf_transfer", new String[] { "hf_transfer" }),
            entry("jinja", new String[] { "Jinja2" }), entry("keras", new String[] { "keras" }),
            entry("minijinja", new String[] { "minijinja" }), entry("numpy", new String[] { "numpy" }),
            entry("pillow", new String[] { "Pillow" }), entry("pydantic", new String[] { "pydantic" }),
            entry("pydot", new String[] { "pydot" }), entry("safetensors", new String[] { "safetensors" }),
            entry("tensorboard", new String[] { "tensorboardX" }),
            entry("tensorflow",
                    new String[] { "tensorflow", "tensorflow-cpu", "tensorflow-gpu", "tf-nightly", "tf-nightly-cpu",
                            "tf-nightly-gpu", "intel-tensorflow", "intel-tensorflow-avx512", "tensorflow-rocm",
                            "tensorflow-macos" }),
            entry("torch", new String[] { "torch" }));

    static {
        _CANDIDATES.forEach((candidate_name, package_names) -> {
            for (String candidate : package_names) {
                try {
                    _package_versions.put(candidate, Package.getPackage(candidate).getImplementationVersion());
                } catch (Exception e) {
                    // pass
                }
            }
        });
    }

    private static String _get_version(String package_name) {
        return _package_versions.getOrDefault(package_name, "N/A");
    }

    private static boolean is_package_available(String package_name) {
        return !"N/A".equals(_get_version(package_name));
    }

    // Python
    public static String get_java_version() {
        return System.getProperty("java.version");
    }

    // Huggingface Hub
    public static String get_hf_hub_version() {
        return Runtime.class.getPackage().getImplementationVersion();
    }

    // FastAI
    public static boolean is_fastai_available() {
        return is_package_available("fastai");
    }

    public static String get_fastai_version() {
        return _get_version("fastai");
    }

    // Fastcore
    public static boolean is_fastcore_available() {
        return is_package_available("fastcore");
    }

    public static String get_fastcore_version() {
        return _get_version("fastcore");
    }

    // Tensorflow
    public static boolean is_tf_available() {
        return is_package_available("tensorflow");
    }

    public static String get_tf_version() {
        return _get_version("tensorflow");
    }

    // Torch
    public static boolean is_torch_available() {
        return is_package_available("torch");
    }

    public static String get_torch_version() {
        return _get_version("torch");
    }
}
