// server.js

const http = require('http');

const PORT = 3333;

const server = http.createServer((req, res) => {
  // 라이츄 API
  if (req.url === '/raichu' && req.method === 'GET') {
    res.writeHead(200, {
      'Content-Type': 'application/json; charset=utf-8',
    });

    res.end(
      JSON.stringify({
        name: '라이츄',
        type: '전기',
        message: '라이츄!',
      }),
    );

    return;
  }

  // 기본 API
  if (req.url === '/' && req.method === 'GET') {
    res.writeHead(200, {
      'Content-Type': 'text/plain; charset=utf-8',
    });

    res.end('Hello World!');
    return;
  }

  // 없는 API
  res.writeHead(404, {
    'Content-Type': 'text/plain; charset=utf-8',
  });

  res.end('Not Found');
});

server.listen(PORT, () => {
  console.log(`Server running at http://localhost:${PORT}`);
});
