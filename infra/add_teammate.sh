#!/usr/bin/env bash
# Create an IAM user for a teammate that can only assume FireLakehouseRole.
# Usage: infra/add_teammate.sh <admin-profile> <username>
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: $0 <admin-profile> <username>" >&2
  exit 2
fi
PROFILE=$1 USERNAME=$2 REGION=us-east-1
ROLE_ARN=$(aws cloudformation describe-stacks --stack-name fire-lakehouse --region "$REGION" --profile "$PROFILE" \
  --query "Stacks[0].Outputs[?OutputKey=='RoleArn'].OutputValue" --output text)

aws iam create-user --user-name "$USERNAME" --profile "$PROFILE" >/dev/null
aws iam add-user-to-group --group-name fire-lakehouse-team --user-name "$USERNAME" --profile "$PROFILE"
read -r KEY_ID SECRET < <(aws iam create-access-key --user-name "$USERNAME" --profile "$PROFILE" \
  --query "AccessKey.[AccessKeyId,SecretAccessKey]" --output text)

cat <<MSG
Created $USERNAME in fire-lakehouse-team.

WARNING: this secret is shown once. Send it to $USERNAME through a private channel
(password-manager share, Signal). Never commit it or post it in a group chat.

--- ~/.aws/credentials ---
[fire-lakehouse-user]
aws_access_key_id = $KEY_ID
aws_secret_access_key = $SECRET

--- ~/.aws/config ---
[profile fire-lakehouse-user]
region = us-east-1

[profile fire-lakehouse]
role_arn = $ROLE_ARN
source_profile = fire-lakehouse-user
region = us-east-1

Then: export AWS_PROFILE=fire-lakehouse
MSG
