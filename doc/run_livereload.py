"""
run `python run_livereload.py` to autocompile docs if changes where made to
the source files.
"""
from livereload import Server, shell

if __name__ == '__main__':
    server2 = Server()
    server2.watch('source/*.rst', shell("make latexpdf > dev/null"), delay=1)
    server2.watch('source/*.md', shell("make latexpdf > dev/null"), delay=1)
    server2.watch('source/*.py', shell("make latexpdf > dev/null"), delay=1)
    server2.watch('source/_static/*', shell("make latexpdf > dev/null"), delay=1)
    server2.watch('source/_templates/*', shell("make latexpdf > dev/null"), delay=1)
    server2.serve(root='build/latex')
