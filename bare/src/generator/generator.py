from src.models.llm_engine import LLMEngine


class GeneratorResponseObject:
    def __init__(self, cost: float, response: str = None):
        self.response = "" if response is None else response
        self.cost = cost


class Generator:
    """General class for LLM generators. Model(s) giving proposal(s)."""

    def __init__(
        self,
        model_name: str,
        system_prompt: str = "",
        model_params: dict = None,
    ):
        """Initializes the generator.

        Args:
            model_name (str): The names of the model to use.
            system_prompt (str): The system prompt to use.
            model_params (dict): The parameters to use for the model.

        Returns:
            None
        """
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.model_params = dict() if model_params is None else model_params
        self.engine = LLMEngine(model_name=model_name, system_prompt=system_prompt)
        self.engine.set_params(**self.model_params)

    def generate(self, prompt: str) -> GeneratorResponseObject:
        """Generates responses from the LLM

        Args:
            prompt: the prompt.

        Returns:
            A `GeneratorResponseObject` with proposed response and cost
        """
        llm_response = self.engine.get_response([{"role": "user", "content": prompt}])
        return GeneratorResponseObject(
            cost=llm_response.cost, response=llm_response.response
        )
