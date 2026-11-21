#!/usr/bin/env bash
# Deploy (or update) the fire-lakehouse stack and upload the ingest code bundle.
# Usage: infra/deploy.sh <admin-profile> <budget-email> <landfire-email> [start-year] [end-year]
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "usage: $0 <admin-profile> <budget-email> <landfire-email> [start-year] [end-year]" >&2
  exit 2
fi
PROFILE=$1 BUDGET_EMAIL=$2 LANDFIRE_EMAIL=$3
START=${4:-2002} END=${5:-2025}
REGION=us-east-1 STACK=fire-lakehouse
cd "$(git rev-parse --show-toplevel)"

if command -v cfn-lint >/dev/null; then cfn-lint infra/lakehouse.yaml; fi

aws cloudformation deploy --stack-name "$STACK" --template-file infra/lakehouse.yaml \
  --capabilities CAPABILITY_NAMED_IAM --region "$REGION" --profile "$PROFILE" \
  --parameter-overrides BudgetEmail="$BUDGET_EMAIL" LandfireEmail="$LANDFIRE_EMAIL" \
    StudyStartYear="$START" StudyEndYear="$END"

BUCKET=$(aws cloudformation describe-stacks --stack-name "$STACK" --region "$REGION" --profile "$PROFILE" \
  --query "Stacks[0].Outputs[?OutputKey=='BucketName'].OutputValue" --output text)

# CodeBuild runs whatever is committed at HEAD: commit before deploying code changes.
BUNDLE=$(mktemp -t ingest).zip
git archive --format=zip -o "$BUNDLE" HEAD lakehouse ingest sql requirements-ingest.txt
aws s3 cp "$BUNDLE" "s3://$BUCKET/_code/ingest.zip" --region "$REGION" --profile "$PROFILE"
rm -f "$BUNDLE"

aws cloudformation describe-stacks --stack-name "$STACK" --region "$REGION" --profile "$PROFILE" \
  --query "Stacks[0].Outputs" --output table
