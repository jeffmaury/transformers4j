package dev.transformers4j.transformers;

import dev.transformers4j.PathUtil;
import dev.transformers4j.hub.Constants;
import io.vavr.control.Either;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Map;

import static dev.transformers4j.hub.FileDownload.try_to_load_from_cache;
import static dev.transformers4j.transformers.utils.Hub.cached_file;
import static dev.transformers4j.transformers.utils.Hub.is_offline_mode;

public class DynamicModuleUtils {
    private static Logger LOGGER = LoggerFactory.getLogger(DynamicModuleUtils.class);

    private static String TRANSFORMERS_DYNAMIC_MODULE_NAME = "transformers_modules";

    private static String HF_MODULES_CACHE = System.getenv().getOrDefault("HF_MODULES_CACHE",
            Constants.HF_HOME + File.separatorChar + "modules");

    /**
     * Extracts a class from a module file, present in the local folder or repository of a model.
     *
     * <Tip warning={true}>
     * <p>
     * Calling this function will execute the code in the module file found locally or downloaded from the Hub. It
     * should therefore only be called on trusted repos.
     *
     * </Tip>
     * <p>
     * <p>
     * <p>
     * Args: class_reference (`str`): The full name of the class to load, including its module and optionally its repo.
     * pretrained_model_name_or_path (`str` or `os.PathLike`): This can be either:
     * <p>
     * - a string, the *model id* of a pretrained model configuration hosted inside a model repo on huggingface.co. - a
     * path to a *directory* containing a configuration file saved using the [`~PreTrainedTokenizer.save_pretrained`]
     * method, e.g., `./my_model_directory/`.
     * <p>
     * This is used when `class_reference` does not specify another repo. module_file (`str`): The name of the module
     * file containing the class to look for. class_name (`str`): The name of the class to import in the module.
     * cache_dir (`str` or `os.PathLike`, *optional*): Path to a directory in which a downloaded pretrained model
     * configuration should be cached if the standard cache should not be used. force_download (`bool`, *optional*,
     * defaults to `False`): Whether or not to force to (re-)download the configuration files and override the cached
     * versions if they exist. resume_download (`bool`, *optional*, defaults to `False`): Whether or not to delete
     * incompletely received file. Attempts to resume the download if such a file exists. proxies (`Dict[str, str]`,
     * *optional*): A dictionary of proxy servers to use by protocol or endpoint, e.g., `{'http': 'foo.bar:3128',
     * 'http://hostname': 'foo.bar:4012'}.` The proxies are used on each request. token (`str` or `bool`, *optional*):
     * The token to use as HTTP bearer authorization for remote files. If `True`, will use the token generated when
     * running `huggingface-cli login` (stored in `~/.huggingface`). revision (`str`, *optional*, defaults to `"main"`):
     * The specific model version to use. It can be a branch name, a tag name, or a commit id, since we use a git-based
     * system for storing models and other artifacts on huggingface.co, so `revision` can be any identifier allowed by
     * git. local_files_only (`bool`, *optional*, defaults to `False`): If `True`, will only try to load the tokenizer
     * configuration from local files. repo_type (`str`, *optional*): Specify the repo type (useful when downloading
     * from a space for instance). code_revision (`str`, *optional*, defaults to `"main"`): The specific revision to use
     * for the code on the Hub, if the code leaves in a different repository than the rest of the model. It can be a
     * branch name, a tag name, or a commit id, since we use a git-based system for storing models and other artifacts
     * on huggingface.co, so `revision` can be any identifier allowed by git.
     *
     * <Tip>
     * <p>
     * Passing `token=True` is required when you want to use a private model.
     *
     * </Tip>
     * <p>
     * Returns: `typing.Type`: The class, dynamically imported from the module.
     * <p>
     * Examples:
     * <p>
     * ```python # Download module `modeling.py` from huggingface.co and cache then extract the class `MyBertModel` from
     * this # module. cls = get_class_from_dynamic_module("modeling.MyBertModel", "sgugger/my-bert-model")
     * <p>
     * # Download module `modeling.py` from a given repo and cache then extract the class `MyBertModel` from this #
     * module. cls = get_class_from_dynamic_module("sgugger/my-bert-model--modeling.MyBertModel",
     * "sgugger/another-bert-model") ```
     */
    public static Class<? extends PretrainedConfig> get_class_from_dynamic_module(String class_reference,
            Path pretrained_model_name_or_path, Path cache_dir, Boolean force_download, Boolean resume_download,
            Map<String, String> proxies, Either<Boolean, String> token, String revision, Boolean local_files_only,
            String repo_type, String code_revision, Map<String, Object> kwargs) {
        throw new UnsupportedOperationException("Not implemented yet");
    }
}
