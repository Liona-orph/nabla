# testdata

Fixtures checked into the repository so that nothing in this project downloads
anything at runtime. A test or an example that reaches the network is a test
that fails on an aeroplane, fails behind a proxy, and one day fails because
somebody else re-organised their web server.

## `sonnets.txt`

*Shake-speares Sonnets*, first published 1609. The work is in the public domain
worldwide; this copy was taken from Project Gutenberg ebook 1041, with the
Project Gutenberg header, footer and licence text removed so that only the
public-domain work remains.

Transformed on the way in, deliberately and once, so that the example does not
have to do it at load time:

* Typographic quotes and dashes folded to ASCII (`'`, `"`, `--`).
* Everything else outside ASCII dropped.
* CRLF folded to LF, and runs of three or more blank lines collapsed to one.

The result is 96,210 characters over a 62-character alphabet, which is the size
that lets `examples/char_transformer` finish on a CPU in a couple of minutes.
