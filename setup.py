from setuptools import find_packages, setup


def get_requirements(file_path='requirements.txt'):
    """
    Reads the requirements.txt file and returns a list of dependencies.

    Parameters:
    - file_path: str, path to the requirements.txt file (default: 'requirements.txt')

    Returns:
    - List of dependencies
    """
    HYPHEN_E_DOT = '-e .'
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
           # requirements = [line.strip() for line in file.readlines() if line.strip() and not line.startswith('#')]
           requirements = file.readlines()
           requirements = [line.replace("\n", "") for line in requirements]

           if HYPHEN_E_DOT in requirements:
               requirements.remove(HYPHEN_E_DOT)
        return requirements
    except FileNotFoundError:
        print(f"Error: {file_path} not found.")
        return []

setup(
    name='Gold Regression',
    version="0.0.1",
    author='Anand Talware',
    author_email="anandtalware27@gmail.com",
    packages=find_packages(),
    install_requires = get_requirements()

)