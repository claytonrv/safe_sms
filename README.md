# safe_sms
A SMS spam classifier created for the Data Mining class on UNIVERSIDADE FEDERAL DE SANTA CATARINA

The project aims to be a tool for SMS classification on the server side.


## How to build?

To build the safe_sms library, enter the project root and paste the following command:

``` bash
    python setup.py sdist bdist_wheel
```

To check if the build worked as expected, you can run the following command:

```bash
    twine check dist/*
```

## How to install the lib?

To install the lib you need first to build the library. Once it's built, run the following command to install:

```bash
    pip install dist/safe_sms-<current_version>-py3-none-any.whl
```

once it's installed, you can open python terminal and import the lib by typing:

```python
    from safe_sms import classifier
    classifier.predict("that's an text example")
```