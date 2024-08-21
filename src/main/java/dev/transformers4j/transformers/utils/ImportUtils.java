package dev.transformers4j.transformers.utils;

import io.vavr.Tuple;
import io.vavr.Tuple2;
import org.semver4j.Semver;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Arrays;
import java.util.List;
import java.util.Set;

public class ImportUtils {
    private static final Logger LOGGER = LoggerFactory.getLogger(ImportUtils.class);

    static Tuple2<Boolean, String> _is_package_available(String pkg_name, boolean return_version) {
        var package_exists = Arrays.stream(Package.getPackages()).filter(p -> p.getName().equals(pkg_name)).findFirst();
        var package_version = "N/A";
        if (package_exists.isPresent()) {
            package_version = package_exists.get().getImplementationVersion();
        }
        return Tuple.of(package_exists.isPresent(), package_version);
    }

    static Package find_spec(String pkg_name) {
        return Arrays.stream(Package.getPackages()).filter(p -> pkg_name.equals(p.getImplementationTitle())).findFirst()
                .orElse(null);
    }

    public static final List<String> ENV_VARS_TRUE_VALUES = Arrays.asList("1", "ON", "YES", "TRUE");
    public static final List<String> ENV_VARS_TRUE_AND_AUTO_VALUES = Arrays.asList("1", "ON", "YES", "TRUE", "AUTO");

    private static final String USE_TF = System.getenv().getOrDefault("USE_TF", "AUTO").toUpperCase();
    private static final String USE_TORCH = System.getenv().getOrDefault("USE_TORCH", "AUTO").toUpperCase();

    private static final String FORCE_TF_AVAILABLE = System.getenv().getOrDefault("FORCE_TF_AVAILABLE", "AUTO")
            .toUpperCase();

    public static String _torch_version = "N/A";
    private static boolean _torch_available = false;

    static {
        if (ENV_VARS_TRUE_AND_AUTO_VALUES.contains(USE_TORCH) && !ENV_VARS_TRUE_VALUES.contains(USE_TF)) {
            var result = _is_package_available("torch", true);
            _torch_available = result._1();
            _torch_version = result._2();

        } else {
            LOGGER.info("Disabling PyTorch because USE_TF is set");
            _torch_available = false;
        }
    }

    public static String _tf_version = "N/A";
    private static boolean _tf_available = false;

    static {
        if (ENV_VARS_TRUE_VALUES.contains(FORCE_TF_AVAILABLE)) {
            _tf_available = true;
        } else {
            if (ENV_VARS_TRUE_AND_AUTO_VALUES.contains(USE_TF) && !ENV_VARS_TRUE_VALUES.contains(USE_TORCH))
                // Note: _is_package_available("tensorflow") fails for tensorflow-cpu. Please test any changes to the
                // line below
                // with tensorflow-cpu to make sure it still works!
                _tf_available = find_spec("tensorflow") != null;
            if (_tf_available) {
                var candidates = Set.of("tensorflow", "tensorflow-cpu", "tensorflow-gpu", "tf-nightly",
                        "tf-nightly-cpu", "tf-nightly-gpu", "tf-nightly-rocm", "intel-tensorflow",
                        "intel-tensorflow-avx512", "tensorflow-rocm", "tensorflow-macos", "tensorflow-aarch64");
                _tf_version = null;
                // For the metadata, we have to look for both tensorflow and tensorflow-cpu
                for (var pkg : candidates) {
                    var result = _is_package_available(pkg, true);
                    if (result._1()) {
                        _tf_version = result._2();
                        break;
                    }
                }
                _tf_available = _tf_version != null;
                if (_tf_available) {
                    if (Semver.parse(_tf_version).isLowerThan("2")) {
                        LOGGER.info(String.format(
                                "TensorFlow found but with version %s. Transformers requires version 2 minimum.",
                                _tf_version));
                        _tf_available = false;
                    }
                }
            } else {
                LOGGER.info("Disabling Tensorflow because USE_TORCH is set");
            }
        }
    }

    public static boolean is_torch_available() {
        return _torch_available;
    }

    public static boolean is_tf_available() {
        return _tf_available;
    }

    public static boolean is_training_run_on_sagemaker() {
        return System.getenv().containsKey("SAGEMAKER_JOB_NAME");
    }

}
