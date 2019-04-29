import http.server
import urllib.request

from main import KEEP

genotype = list()
score = list()

PORT = 8888


class Controller(http.server.BaseHTTPRequestHandler):
    def __init__(self, a, b, c):
        http.server.BaseHTTPRequestHandler.__init__(self, a, b, c)

    def do_POST(self):
        """Respond to a GET request."""
        received = self.rfile.read(int(self.headers.get("Content-Length"))).decode("utf-8")
        for l in received.splitlines():
            [val, gen] = l.split(" ", 1)
            val = float(val)
            if len(genotype) < KEEP:
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
        self.do_GET()

    def do_GET(self):
        """Respond to a GET request."""
        self.send_response(200)
        self.send_header("Content-type", "application/octet-stream")
        self.end_headers()
        for i in range(len(genotype)):
            self.wfile.write(bytes(str(score[i]) + " " + genotype[i], "utf-8"))
            self.wfile.write(b"\n")


def get_top(remote: str):
    res = (
        urllib.request.urlopen("http://{}:{}/".format(remote, PORT), timeout=5)
            .read()
            .decode("utf-8")
    )
    return res.splitlines()


def post_top(remote: str, best):
    concatenated = [str(i[0]) + " " + i[1] for i in best]
    data = "\n".join(concatenated)
    raw_res = urllib.request.urlopen(
        "http://{}:{}/".format(remote, PORT), bytes(data, "utf-8"), timeout=5
    )
    raw_res = raw_res.read().decode("utf-8")
    listed_res = []
    for e in raw_res.splitlines():
        [val, gen] = e.split(" ", 1)
        val = float(val)
        listed_res.append((val, gen))
    return listed_res


if __name__ == "__main__":
    server_address = ("", PORT)
    server = http.server.HTTPServer
    handler = Controller
    httpd = server(server_address, handler)
    httpd.serve_forever()
