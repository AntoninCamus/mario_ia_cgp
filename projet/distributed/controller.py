import http.server
import urllib.request

keep = 5
genotype = list()
score = list()

PORT = 8888


class Controller(http.server.BaseHTTPRequestHandler):
    def __init__(self, a, b, c):
        http.server.BaseHTTPRequestHandler.__init__(self, a, b, c)

    def do_POST(s):
        """Respond to a GET request."""
        received = s.rfile.read1().decode("utf-8")
        for l in received.splitlines():
            [val, gen] = l.split(" ")
            val = int(val)
            if len(genotype) < keep:
                score.append(val)
                genotype.append(gen)
                continue
            m = min(score)
            if m < val:
                i = score.index(m)
                score[i] = val
                genotype[i] = gen
                print("New top genotype:")
                for g in genotype:
                    print(g)
                print("")

        s.do_GET()

    def do_GET(s):
        """Respond to a GET request."""
        s.send_response(200)
        s.send_header("Content-type", "application/octet-stream")
        s.end_headers()
        for i in range(len(genotype)):
            s.wfile.write(bytes(genotype[i], "utf-8"))
            s.wfile.write(b"\n")


def get_top(remote: str):
    res = urllib.request.urlopen("http://" + remote + "/", timeout=5).read().decode("utf-8")
    return res.splitlines()


def post_top(remote: str, best):
    conc = [str(i[0]) + " " + i[1] for i in best]
    data = "\n".join(conc)
    res = urllib.request.urlopen("http://{}:{}/".format(remote, PORT), bytes(data, "utf-8"), timeout=5)
    res = res.read().decode("utf-8")
    return res.splitlines()


if __name__ == "__main__":
    server_address = ("", PORT)
    server = http.server.HTTPServer
    handler = Controller
    httpd = server(server_address, handler)
    httpd.serve_forever()
