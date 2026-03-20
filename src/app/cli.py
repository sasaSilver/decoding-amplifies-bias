from pydantic_settings import CliApp

from .settings.settings import Settings

def main() -> None:
    CliApp.run(Settings)

if __name__ == "__main__":
    main()