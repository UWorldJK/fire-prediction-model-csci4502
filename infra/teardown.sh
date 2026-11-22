#!/usr/bin/env bash
# PERMANENTLY delete the lakehouse: all data, teammate IAM users, and the stack.
# Usage: infra/teardown.sh <admin-profile>
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <admin-profile>" >&2
  exit 2
fi
PROFILE=$1 REGION=us-east-1 STACK=fire-lakehouse
BUCKET=$(aws cloudformation describe-stacks --stack-name "$STACK" --region "$REGION" --profile "$PROFILE" \
  --query "Stacks[0].Outputs[?OutputKey=='BucketName'].OutputValue" --output text)

aws s3 ls "s3://$BUCKET" --recursive --summarize --human-readable --profile "$PROFILE" | tail -2
echo "This deletes EVERYTHING in s3://$BUCKET (bronze and silver), all users in"
echo "fire-lakehouse-team, and the $STACK stack. It cannot be undone."
read -r -p "Type the bucket name to confirm: " CONFIRM
if [[ "$CONFIRM" != "$BUCKET" ]]; then
  echo "Bucket name did not match; nothing deleted." >&2
  exit 1
fi

aws s3 rm "s3://$BUCKET" --recursive --profile "$PROFILE"

for USER in $(aws iam get-group --group-name fire-lakehouse-team --profile "$PROFILE" \
    --query "Users[].UserName" --output text); do
  for KEY in $(aws iam list-access-keys --user-name "$USER" --profile "$PROFILE" \
      --query "AccessKeyMetadata[].AccessKeyId" --output text); do
    aws iam delete-access-key --user-name "$USER" --access-key-id "$KEY" --profile "$PROFILE"
  done
  aws iam remove-user-from-group --group-name fire-lakehouse-team --user-name "$USER" --profile "$PROFILE"
  aws iam delete-user --user-name "$USER" --profile "$PROFILE"
  echo "deleted user $USER"
done

aws cloudformation delete-stack --stack-name "$STACK" --region "$REGION" --profile "$PROFILE"
aws cloudformation wait stack-delete-complete --stack-name "$STACK" --region "$REGION" --profile "$PROFILE"
echo "Stack $STACK deleted. Remember to delete the fire-lakehouse-admin access key in IAM."
