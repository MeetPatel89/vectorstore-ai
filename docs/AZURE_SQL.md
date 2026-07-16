# Azure SQL vector store setup

`AzureSqlVectorStore` stores chunks in Azure SQL's native `VECTOR` type and
ranks them with exact server-side cosine distance. It uses Microsoft's
`mssql-python` DB-API driver and opens a fresh connection for every operation.
Writes are transactional, and values are passed as query parameters.

## Engine and dimension requirements

Use Azure SQL Database, or Azure SQL Managed Instance with the SQL Server 2025
or Always-up-to-date update policy. The stable float32 `VECTOR` type accepts
dimensions from 1 through 1,998. The default `text-embedding-3-small` width of
1,536 fits. Configure larger embedding models to return no more than 1,998
dimensions before creating the table.

The backend uses exact `VECTOR_DISTANCE` queries. Microsoft recommends exact
search when each query evaluates fewer than roughly 50,000 vectors, including
after metadata predicates. Approximate vector indexes and `VECTOR_SEARCH` are
still preview features, have region and deployment constraints, and are not
enabled by this backend.

References:

- [Azure SQL vector data type and limits](https://learn.microsoft.com/en-us/sql/t-sql/data-types/vector-data-type?view=azuresqldb-current)
- [Exact and approximate vector search](https://learn.microsoft.com/en-us/sql/sql-server/ai/vectors?view=azuresqldb-current)
- [`VECTOR_DISTANCE` behavior](https://learn.microsoft.com/en-us/sql/t-sql/functions/vector-distance-transact-sql?view=azuresqldb-current)

## Install the driver

```bash
uv sync --extra azure-sql
```

`mssql-python` is Microsoft's supported Python driver and includes Microsoft
Entra authentication. Linux and macOS hosts can require the system libraries
listed in Microsoft's installation instructions.

- [mssql-python driver quickstart](https://learn.microsoft.com/en-us/sql/connect/python/mssql-python/python-sql-driver-mssql-python-quickstart?view=sql-server-ver17)

## Configure passwordless authentication

First configure a Microsoft Entra administrator on the Azure SQL logical server.
For local development, sign in with `az login`, then use
`ActiveDirectoryDefault`:

```bash
export AZURE_SQL_CONNECTIONSTRING="Server=<server>.database.windows.net;Database=<database>;Authentication=ActiveDirectoryDefault;Encrypt=yes;TrustServerCertificate=no;"
```

For a system-assigned managed identity hosted in Azure, first enable the
identity on the App Service, Function, Container App, VM, or other host, then
use:

```text
Server=<server>.database.windows.net;Database=<database>;Authentication=ActiveDirectoryMSI;Encrypt=yes;TrustServerCertificate=no;
```

For a user-assigned managed identity, add its client ID:

```text
Server=<server>.database.windows.net;Database=<database>;Authentication=ActiveDirectoryMSI;User Id=<managed-identity-client-id>;Encrypt=yes;TrustServerCertificate=no;
```

Keep the connection string in an application setting or secret configuration,
not source control. These passwordless strings contain no database password.

- [Configure Microsoft Entra authentication for Azure SQL](https://learn.microsoft.com/en-us/azure/azure-sql/database/authentication-aad-configure?view=azuresql)
- [Azure SQL Python passwordless quickstart](https://learn.microsoft.com/en-us/azure/azure-sql/database/azure-sql-python-quickstart?view=azuresql)

## Provision identities with least privilege

Use separate deployment and runtime identities. As the Microsoft Entra database
administrator, create contained database users for both identities:

```sql
CREATE USER [vectorstore-deployer] FROM EXTERNAL PROVIDER;
CREATE USER [vectorstore-app] FROM EXTERNAL PROVIDER;
```

Use a dedicated schema so the deployment identity does not need `ALTER` on
`dbo`. As an administrator, create the schema, then grant the deployment
identity `CREATE TABLE` on the database and `ALTER` only on that schema:

```sql
CREATE SCHEMA [vectorstore] AUTHORIZATION [dbo];
GRANT CREATE TABLE TO [vectorstore-deployer];
GRANT ALTER ON SCHEMA::[vectorstore] TO [vectorstore-deployer];
```

Connect as that identity and create the table once:

```python
from vectorstore import AzureSqlVectorStore

AzureSqlVectorStore(
    dimension=1536,
    schema_name="vectorstore",
    initialize_schema=True,
)
```

If a separate migration system owns schema changes, use its connection to run
`store.schema_sql` instead. Do not grant DDL permissions to the application
identity.

After the table exists, grant only object-level runtime access through a custom
role:

```sql
CREATE ROLE [vectorstore_runtime];
GRANT SELECT, INSERT, UPDATE, DELETE
    ON OBJECT::[vectorstore].[vectorstore_chunks]
    TO [vectorstore_runtime];
ALTER ROLE [vectorstore_runtime] ADD MEMBER [vectorstore-app];
```

The runtime store uses schema initialization's default of `False`:

```python
from vectorstore import AzureSqlVectorStore

store = AzureSqlVectorStore(dimension=1536, schema_name="vectorstore")
store.validate_schema()
```

Azure resource roles such as Contributor manage the Azure resource but do not
grant SQL data-plane access. The contained database user and T-SQL grants above
are still required.

Creating Microsoft Entra users can require the Azure SQL server identity to
read the relevant principal in Microsoft Graph. Azure SQL Database supports
fine-grained `Application.Read.All`, `User.Read.All`, or
`GroupMember.Read.All`, depending on the principal type. SQL Managed Instance
can require Directory Readers. A tenant administrator must configure these
directory permissions; the application identity does not need them for normal
database access.

## Configure network access

Authentication and network reachability are separate. Prefer a private endpoint
with private DNS, approve the endpoint connection, and disable public network
access after validating the private path. If a public endpoint is necessary,
allow only the application's outbound IP addresses or subnet.

Avoid the broad **Allow Azure services and resources to access this server**
rule for production. It permits network attempts from Azure resources outside
your subscription; database authentication still applies, but the network
boundary is much wider.

- [Azure SQL private endpoints](https://learn.microsoft.com/en-us/azure/azure-sql/database/private-endpoint-overview?view=azuresql)
- [Azure SQL firewall rules](https://learn.microsoft.com/en-us/azure/azure-sql/database/firewall-configure?view=azuresql)

## Use a custom table

Schema and table names are configurable. They are restricted to simple SQL
identifiers to prevent identifier injection:

```python
store = AzureSqlVectorStore(
    dimension=1536,
    schema_name="search",
    table_name="support_chunks",
)
```

The table stores a case-sensitive string ID, chunk text, JSON metadata, and one
float32 vector. Metadata equality, `$in`, `$gt`, `$gte`, `$lt`, and `$lte`
filters are applied in SQL before `TOP (k)`. Upserts use update locks and a
serializable key range so concurrent inserts of the same ID do not create
duplicates.

## Troubleshooting

- **Login failed / principal not found:** confirm the Microsoft Entra admin,
  contained user, managed identity client ID, and target database.
- **Connection timeout:** confirm DNS, private endpoint approval, routing, and
  firewall rules before changing authentication settings.
- **Table missing:** run schema bootstrap with the deployment identity; keep
  `initialize_schema=False` for the runtime identity.
- **Dimension mismatch:** the configured dimension, embedding provider output,
  and existing table `VECTOR(n)` must be identical. Recreate or migrate the
  table deliberately rather than silently mixing embedding spaces.
- **Driver import failure on Linux/macOS:** install the platform dependencies
  from the Microsoft driver quickstart, then rerun `uv sync --extra azure-sql`.
