from livereload import Server, shell

if __name__ == '__main__':
#     server = Server()
#     server.watch('source/*.rst', shell("make html"), delay=1)
#     server.watch('source/*.md', shell("make html"), delay=1)
#     server.watch('source/*.py', shell("make html"), delay=1)
#     server.watch('source/_static/*', shell("make html"), delay=1)
#     server.watch('source/_templates/*', shell("make html"), delay=1)
#     server.serve(root='build/html')
    
    server2 = Server()
    server2.watch('source/*.rst', shell("make latexpdf > dev/null"), delay=1)
    server2.watch('source/*.md', shell("make latexpdf > dev/null"), delay=1)
    server2.watch('source/*.py', shell("make latexpdf > dev/null"), delay=1)
    server2.watch('source/_static/*', shell("make latexpdf > dev/null"), delay=1)
    server2.watch('source/_templates/*', shell("make latexpdf > dev/null"), delay=1)
    server2.serve(root='build/latex')
