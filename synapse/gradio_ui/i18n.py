"""
Synapse Music V12 - Internationalization (i18n) Module

Provides translation management for the Synapse Music Gradio interface.
Supports multiple languages through JSON translation files stored in the
local i18n directory.
"""

import json
import os
from typing import Dict, Optional


class I18n:
    """
    Synapse Music internationalization handler.

    Loads language-specific JSON translation files and provides nested
    translation-key lookup with English fallback support.
    """

    def __init__(self, default_language: str = "en"):
        """
        Initialize the Synapse Music internationalization handler.

        Args:
            default_language:
                Default language code such as "en", "zh", or "ja".
        """
        self.current_language = default_language
        self.translations: Dict[str, Dict] = {}

        self._load_all_translations()

    def _load_all_translations(self) -> None:
        """
        Load all JSON translation files from the local i18n directory.

        Translation files are expected to use their language code as the
        filename, for example:

            en.json
            zh.json
            ja.json
        """
        current_file = os.path.abspath(__file__)
        module_dir = os.path.dirname(current_file)
        i18n_dir = os.path.join(module_dir, "i18n")

        if not os.path.exists(i18n_dir):
            os.makedirs(
                i18n_dir,
                exist_ok=True,
            )
            return

        for filename in sorted(os.listdir(i18n_dir)):
            if not filename.endswith(".json"):
                continue

            lang_code = filename[:-5]
            filepath = os.path.join(
                i18n_dir,
                filename,
            )

            try:
                with open(
                    filepath,
                    "r",
                    encoding="utf-8",
                ) as file:
                    translation_data = json.load(file)

                if not isinstance(translation_data, dict):
                    print(
                        f"Warning: Synapse translation file "
                        f"'{filename}' does not contain a JSON object."
                    )
                    continue

                self.translations[lang_code] = translation_data

            except json.JSONDecodeError as error:
                print(
                    f"Error parsing Synapse translation file "
                    f"'{filename}': {error}"
                )

            except OSError as error:
                print(
                    f"Error reading Synapse translation file "
                    f"'{filename}': {error}"
                )

            except Exception as error:
                print(
                    f"Error loading Synapse translation file "
                    f"'{filename}': {error}"
                )

    def set_language(self, language: str) -> None:
        """
        Set the active Synapse Music interface language.

        Args:
            language:
                Language code to activate.
        """
        if language in self.translations:
            self.current_language = language
            return

        print(
            f"Warning: Synapse language '{language}' was not found. "
            f"Keeping '{self.current_language}'."
        )

    def t(self, key: str, **kwargs) -> str:
        """
        Translate a Synapse Music interface key.

        Args:
            key:
                Translation key using dot-separated nested notation.

                Example:
                    "generation.generate_btn"

            **kwargs:
                Optional formatting parameters used by the translated string.

        Returns:
            The translated string.

            Lookup order:

            1. Current language
            2. English
            3. Translation key itself
        """
        translation = self._get_nested_value(
            self.translations.get(
                self.current_language,
                {},
            ),
            key,
        )

        if translation is None:
            translation = self._get_nested_value(
                self.translations.get(
                    "en",
                    {},
                ),
                key,
            )

        if translation is None:
            translation = key

        if kwargs:
            try:
                translation = translation.format(
                    **kwargs
                )
            except (KeyError, IndexError, ValueError):
                pass

        return translation

    def _get_nested_value(
        self,
        data: dict,
        key: str,
    ) -> Optional[str]:
        """
        Retrieve a nested translation value using dot notation.

        Args:
            data:
                Translation dictionary to search.

            key:
                Dot-separated translation key.

                Example:
                    "results.generated_music"

        Returns:
            The translated string when found, otherwise None.
        """
        keys = key.split(".")
        current = data

        for nested_key in keys:
            if (
                isinstance(current, dict)
                and nested_key in current
            ):
                current = current[nested_key]
            else:
                return None

        if isinstance(current, str):
            return current

        return None

    def get_available_languages(self) -> list:
        """
        Return all available Synapse Music language codes.

        Returns:
            List of loaded language codes.
        """
        return list(
            self.translations.keys()
        )


# =========================================================================
# Global Synapse Music i18n Instance
# =========================================================================

_i18n_instance: Optional[I18n] = None


def get_i18n(
    language: Optional[str] = None,
) -> I18n:
    """
    Return the global Synapse Music internationalization instance.

    Args:
        language:
            Optional language code to activate.

    Returns:
        Global I18n instance.
    """
    global _i18n_instance

    if _i18n_instance is None:
        _i18n_instance = I18n(
            default_language=language or "en"
        )

    elif language is not None:
        _i18n_instance.set_language(
            language
        )

    return _i18n_instance


def t(
    key: str,
    **kwargs,
) -> str:
    """
    Translate a Synapse Music interface key using the global i18n instance.

    Args:
        key:
            Translation key.

        **kwargs:
            Optional formatting parameters.

    Returns:
        Translated interface string.
    """
    return get_i18n().t(
        key,
        **kwargs,
    )