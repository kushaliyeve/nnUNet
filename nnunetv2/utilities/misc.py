import copy
import importlib
import logging
from typing import Union
from uuid import uuid4

# copied from ai_ovai repo

logger = logging.getLogger(__name__)


def generate_id() -> str:
    """Generate a unique id.

    Returns:
        str: a UUID v4 based id (separated by dashes)
    """
    return str(uuid4())


class RunName:
    """Class that generates a run name for the current simulation run."""

    def __init__(self, dictionary: dict, default_run_name: str = None) -> None:
        """
        Args:
            dictionary (dict): input configuration dictionary.
            default_run_name (str, optional): default name to be assigned when no parameters can be extracted from the string. Defaults to None.
        """
        # make a deep copy since generate() could modify it for formatting purposes
        # but we don't want to make these changes visible to the client
        self.dictionary = copy.deepcopy(dictionary)
        self.default_run_name = default_run_name

    def generate(
        self,
        skip_keys=[],
        max_string_len: int = 12,
        max_number_digits: int = 4,
        scientific_notation_decimals: int = 2,
    ):
        """Function to generate a valid run name (string) from key parameters provided from dictionary

        This function extracts keys and values from the provided configuration dictionary and concatenates keys
        and values to build a custom string rapresentative of the input configuration.
        If the input dictionary is empty, or if none key is valid, the name used for the naming
        is the default run name.
        Dictionary keys can be selectively skipped from the output run name.
        Dictionary values are formatted according to speficic format rules depending  on their type (see private methods for details);

        - Managed values: int, float, list, None, str,bool
        - Always skipped values: dictionary, Paths (more precisely strings containing "/")

        Args:
            skip_keys (list, optional): keys to be skipped from the resulting run name. Defaults to [].
            max_string_len (int, optional): Max len of the string values to concatenate. Defaults to 12.
            max_number_digits (int, optional): Max number of digits allowed before the forced computation of scientific notation number. Defaults to 4.
            scientific_notation_decimals (int, optional): Number of decimals to be set within the scientific notation. Defaults to 2.

        Returns:
            run_name(str): final run name
        """

        # if the dictionary is empty OR all the keys of the dictionaries are skipped (no valid key remains for naming) --> return default_run_name
        if not (set(self.dictionary.keys()).difference(set(skip_keys))):
            return self.default_run_name

        else:
            # get rid of skipping parameters (ex CV_seed and CV_fold)
            [self.dictionary.pop(skip_key) for skip_key in skip_keys if skip_key in self.dictionary]

            # sort the keys to mantain an overall order of the parameters
            self.dictionary = dict(sorted(self.dictionary.items()))

            # empty string as run_name for further formatting
            run_name = ""

            for key, value in self.dictionary.items():
                if isinstance(value, (list, float, int, str, bool)) or value is None:
                    current_value = self._generic_value_formatting(
                        value,
                        max_string_len,
                        max_number_digits,
                        scientific_notation_decimals,
                    )

                    # only if the value can be correctly formatted and is given as output (ex: paths are string, but are skipped within string formatting)
                    if current_value is not None:
                        parameter_name_to_concatenate_and_value = f"{key}_{current_value}"

                        # apply join if no parameter was concatenated before, else assign that value to run_name (first parameter+value)
                        run_name = (
                            parameter_name_to_concatenate_and_value
                            if run_name == ""
                            else ("_").join([run_name, parameter_name_to_concatenate_and_value])
                        )

            # if no valid parameters were added for the naming, return default_run_name else return the valid name
            return self.default_run_name if (run_name == "") else run_name

    @staticmethod
    def _format_str(input_string: str, max_string_len: int = 12):
        """Format string according to a specific set of rules.

        Rules:
        - The output string will be always lowered and shortified (see shoritify function)
        - If the input string does not contain any letter or digit, the string is skipped
        - If the string contains at least one '/', it is automatically interpreted as a path
        and therefore excluded as output value
        - If the string contains at least a dot '.', it will be treated as a classpath and in the case
        the module is found, the last piece (separated by ".") will be extracted and formatted according to the
        first rule. In all the other cases the string is skipped.

        Raises:
            ValueError: If the input value is not a string

        Returns:
            str: output formatted string
        """

        if isinstance(input_string, (str)):
            if not any(char.isalpha() or char.isdigit() for char in input_string):
                return None

            if "/" in input_string:  # is a path --> do not accept
                return None

            if "." in input_string:
                try:
                    # Try importing the module dynamically
                    # Split the class path to get module and class names
                    module_name, class_name = input_string.rsplit(".", 1)

                    # Try importing the module dynamically
                    module = importlib.import_module(module_name)

                    # Check if the class exists in the module
                    getattr(module, class_name)

                    # return the class name formatted
                    return shortify(class_name.lower(), max_string_len)

                except (ImportError, ModuleNotFoundError) as e:
                    # Handle ImportError or ModuleNotFoundError (Module not found)
                    print(f"Error importing class path '{input_string}': {e}")
                    return None
                except AttributeError as e:
                    # Handle AttributeError (Class not found within the module)
                    print(f"Error: '{input_string}' is not a valid class path. {e}")
                    return None
                except Exception as e:
                    # Handle other types of exceptions
                    print(
                        f"An unexpected error occurred while checking class path '{input_string}': {e}",
                    )
                    return None
            else:
                # any generic string that does not contain "." or "/"
                return shortify(input_string.lower(), max_string_len)
        else:
            msg = "Input must be a string"
            raise ValueError(msg)

    @staticmethod
    def _format_number(
        input_number: Union[int, float],
        max_digits: int = 4,
        scientific_notation_decimals: int = 2,
    ):
        """Format number into string, according to a specific set of rules.

        Args:
            input_number (int|float): number to format
            max_digits (int, optional): Max number of digits after which apply the scientific notation formatting. Defaults to 4.
            scientific_notation_decimals (int, optional): Number of decimals of the output scientific notation. Defaults to 2.

        Rules:
        - If the number is already with scientific notation, apply the wanted decimals.
        - If the number is not with scientific notation, evaluate max number of digits desired and, in case of exceeding, apply desired scientific notation

        Raises:
            ValueError: If the input value is not a float nor int

        Returns:
            number(str): Output formatted number
        """

        if isinstance(input_number, (int, float)):
            # if we don't have scientific notation already, apply if number of digits is over the provided max_digits
            if "e" not in str(input_number).lower():
                # compute current digits
                current_digits = str(abs(input_number)).replace(".", "")

                # evaluate number of digits
                if len(current_digits) > max_digits:
                    # apply scientific notation with custom numbers of decimals
                    format_string = "{{:.{}e}}".format(scientific_notation_decimals)
                    formatted_number = format_string.format(input_number)
                    return formatted_number
                else:
                    return str(input_number)
            else:
                format_string = "{{:.{}e}}".format(scientific_notation_decimals)
                formatted_number = format_string.format(input_number)
                return formatted_number
        else:
            msg = "Input must be an int or float"
            raise ValueError(msg)

    @staticmethod
    def _format_list(
        input_list: list,
        max_string_len: int = 12,
        max_number_digits: int = 4,
        scientific_notation_decimals: int = None,
    ):
        """Formats input list into string by concatenation of its values (after formatting)."""
        if len(input_list):
            list_str = "_".join(
                [
                    RunName._generic_value_formatting(
                        value,
                        max_string_len,
                        max_number_digits,
                        scientific_notation_decimals,
                    )
                    for value in input_list
                    if RunName._generic_value_formatting(
                        value,
                        max_string_len,
                        max_number_digits,
                        scientific_notation_decimals,
                    )
                ],
            )
        else:
            list_str = "[]"

        return list_str

    @staticmethod
    def _generic_value_formatting(
        value,
        max_string_len: int = 12,
        max_number_digits: int = 4,
        scientific_notation_decimals: int = None,
    ):
        """Functions that associates input value type to the required formatting function.

        Current managed value types: bool, str, int, float, list and 'None' values
        All the other types are skipped and 'None' value is returned.

        Attention: None input value leads to 'null' output str.

        Returns:
            Union[str, None]: value formatted into string or None.
        """

        if value is None:
            return "null"
        elif isinstance(value, bool):
            return str(value)
        elif isinstance(value, str):
            return RunName._format_str(value, max_string_len)
        elif isinstance(value, (int, float)):
            return RunName._format_number(value, max_number_digits, scientific_notation_decimals)
        elif isinstance(value, list):
            return RunName._format_list(
                value,
                max_string_len,
                max_number_digits,
                scientific_notation_decimals,
            )
        else:
            logger.info(f"Current type ({type(value)}) cannot be formatted, skipped")
            return None


def shortify(str: str, max_str_len: int = 10):
    """Shortify a string adding ellipsis ("...") if its lenght is greather than a threshold.

    Only the firs and last max_str_len//2 characters will be preserved in the shortified
    string

    Note:
        This function can be used to improve plots readability with long titles

    Args:
        str (str): input string to shorten
        max_str_len (int, optional): max number of input string characters to show (excluded the "..." ellipsis). Defaults to 10.

    Returns:
        str: shortified string
    """
    return (
        str[: max_str_len // 2] + "..." + str[-max_str_len // 2 :]
        if len(str) > max_str_len
        else str
    )
