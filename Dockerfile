FROM python:3.12-slim

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

COPY . .
RUN uv sync --no-dev

EXPOSE 80

# Data paths are configured at runtime via environment variables:
#   BHUMI_GAIA_DATA_ROOT   (default: /mnt/ceph/users/gaia/dr3/hdf5)
#   BHUMI_ANDRAE_CATALOG   (default: /mnt/home/apricewhelan/data/Gaia/vac/Andrae2023/table_1_catwise.fits)
#   BHUMI_ZHANG_CATALOG    (default: /mnt/home/apricewhelan/data/Gaia/vac/Zhang2023/stellar_params_catalog_joined.h5)

CMD ["uv", "run", "uvicorn", "bhumi.app:app", "--host", "0.0.0.0", "--port", "80"]
