class ConfigurationException(RuntimeError):
    def __init__(self, msg=None):
        if type(self) == ConfigurationException:
            raise NotImplementedError("{} must be subclassed.".format(self.__class__.__name__))
        super().__init__(msg)


class HyperparamsConfigurationException(ConfigurationException):
    def __init__(self, msg=None):
        super().__init__(msg)


class DirectoryNotFoundError(OSError):
    def __init__(self, msg=None):
        super().__init__(msg)


class InvalidFileType(OSError):
    def __init__(self, msg=None):
        super().__init__(msg)


class InvalidDirectoryLocation(OSError):
    def __init__(self, msg=None):
        super().__init__(msg)
