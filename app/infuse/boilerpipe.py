import jpype

from settings import JAVA_CLASSPATH

_jvm_started = False

def transform(content):
    """Transform a HTML content into a readable text, removing all the unwanted
    content (menus, footers, etc)
    """
    start_jvm()
    DefaultExtractor = jpype.JPackage("de").l3s.boilerpipe.extractors.DefaultExtractor
    return DefaultExtractor.INSTANCE.getText(content)

def start_jvm():
    """Ensure the jvm is started.

    This is done using the singleton design pattern
    """
    global _jvm_started
    if not _jvm_started:
        jpype.startJVM(jpype.getDefaultJVMPath(), 
                "-Djava.class.path=%s" % JAVA_CLASSPATH)
        _jvm_started = True

def stop_jvm():
    """Shutdown the JVM if it is started"""
    global _jvm_started
    if _jvm_started:
        jpype.shutdownJVM()
