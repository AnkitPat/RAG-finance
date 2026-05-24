from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import SecretStr

class Settings(BaseSettings):
    GOOGLE_API_KEY: SecretStr
    OPENAI_API_KEY: SecretStr | None = None

    model_config = SettingsConfigDict(env_file=".env")

settings = Settings()
