#!/usr/bin/env bash
# Start a serverless ingest run in CodeBuild and follow its logs.
# Usage: infra/run_ingest.sh <profile> <target> [--medium]
#   target: gridmet | mtbs | landfire | integrate | catalog | viewer | all
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: $0 <profile> <gridmet|mtbs|landfire|integrate|catalog|viewer|all> [--medium]" >&2
  exit 2
fi
PROFILE=$1 TARGET=$2 REGION=us-east-1
EXTRA=()
if [[ ${3:-} == "--medium" ]]; then EXTRA=(--compute-type-override BUILD_GENERAL1_MEDIUM); fi

BUILD_ID=$(aws codebuild start-build --project-name fire-lakehouse-ingest --region "$REGION" --profile "$PROFILE" \
  --environment-variables-override "name=INGEST_TARGET,value=$TARGET,type=PLAINTEXT" "${EXTRA[@]}" \
  --query build.id --output text)
echo "started $BUILD_ID (target=$TARGET)"
echo "status: aws codebuild batch-get-builds --ids $BUILD_ID --query 'builds[0].buildStatus' --profile $PROFILE --region $REGION"
aws logs tail /aws/codebuild/fire-lakehouse-ingest --follow --since 1m --region "$REGION" --profile "$PROFILE"
