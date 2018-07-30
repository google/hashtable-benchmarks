#!/bin/bash

set -e

REV=$(git rev-parse HEAD)
SUFFIX=$(git diff-index --quiet HEAD -- || echo -dirty)

echo "BUILD_SCM_REVISION ${REV}${SUFFIX}"
