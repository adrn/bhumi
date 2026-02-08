# Bhumi

A web viewer for [Gaia DR3](https://www.cosmos.esa.int/web/gaia/dr3) data,
designed to run on a machine with local access to the full Gaia DR3 data stored
as HDF5 files.

Enter any Gaia DR3 `source_id` and get a summary page of the Gaia data, with
additional derived quantities and an orbital analysis if full 6D phase-space
information is available. The orbit is computed on the fly with
[gala](https://gala.adrian.pw/)

Every source page has a shareable URL (`?source_id=...`).

## Requirements

- [uv](https://docs.astral.sh/uv/) for package management
- Local access to Gaia DR3 HDF5 files

## Installation

```sh
git clone https://github.com/adrn/bhumi.git
cd bhumi
uv sync
```

## Usage

### Quick start (foreground)

```sh
uv run uvicorn bhumi.app:app --reload --host 0.0.0.0 --port 8000
```

Then open `http://localhost:8000` in your browser.

### Server management script

A management script is provided for running the server as a background process:

```sh
# Start the server in the background
./bhumi-server.sh start

# Check if the server is running
./bhumi-server.sh status

# View recent logs
./bhumi-server.sh logs

# Follow the log stream
./bhumi-server.sh logs -f

# Restart the server
./bhumi-server.sh restart

# Stop the server
./bhumi-server.sh stop
```

## License

See [LICENSE](LICENSE).
