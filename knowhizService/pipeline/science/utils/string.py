class StringUtil:
    @staticmethod
    def ensure_utf8_encoding(data):
        """
        Recursively ensures all string values in a dictionary are encoded in UTF-8.
        """
        if isinstance(data, dict):
            return {key: StringUtil.ensure_utf8_encoding(value) for key, value in data.items()}
        elif isinstance(data, list):
            return [StringUtil.ensure_utf8_encoding(item) for item in data]
        elif isinstance(data, str):
            return data.encode('utf-8').decode('utf-8')
        else:
            return data