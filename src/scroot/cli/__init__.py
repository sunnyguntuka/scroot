"""Scroot CLI - `scroot serve` starts the review dashboard."""
from __future__ import annotations

try:
    import typer
    app = typer.Typer(name="scroot", help="Scroot LLM response quality tools.")

    @app.command("download-model")
    def download_model_cmd(
        model: str = typer.Option("phi4-mini", help="Model ID to download (phi4-mini, smollm3)"),
    ):
        """Download a local LLM model for offline correction."""
        from scroot.cli.download import download_model
        try:
            download_model(model)
        except Exception as e:
            typer.echo(f"ERROR: {e}")
            raise typer.Exit(1)

    @app.command("model-info")
    def model_info_cmd():
        """List available local LLM models and their download status."""
        from scroot.cli.model_info import print_model_info
        print_model_info()

    @app.command()
    def score(
        query: str = typer.Option(..., "--query", "-q", help="The user's query/question."),
        response: str = typer.Option(..., "--response", "-r", help="The LLM-generated response."),
        context: list[str] = typer.Option(
            None, "--context", "-c", help="Grounding context chunk. Repeat for multiple chunks."
        ),
        json_output: bool = typer.Option(
            False, "--json", help="Print the full result as JSON instead of a summary."
        ),
    ):
        """Score a single query/response pair and print the result."""
        import json as json_module
        from scroot import score as score_fn

        result = score_fn(query=query, response=response, context=context or None)

        if json_output:
            typer.echo(json_module.dumps(result.to_dict(), indent=2, default=str))
            return

        typer.echo(f"IQS:          {result.iqs:.2f}")
        typer.echo(
            f"Groundedness: {result.groundedness:.2f}"
            if result.groundedness is not None
            else "Groundedness: n/a (no context provided)"
        )
        typer.echo(f"Completeness: {result.completeness:.2f}")
        typer.echo(f"Relevance:    {result.relevance:.2f}")
        typer.echo(f"Consistency:  {result.consistency:.2f}")
        typer.echo(f"Confidence:   {result.confidence:.2f}")
        typer.echo(f"Flags:        {result.flags}")

    @app.command("eval")
    def eval_cmd(
        suite: str = typer.Option(..., "--suite", "-s", help="Path to a YAML eval suite."),
        fail_below: float = typer.Option(
            None, "--fail-below", help="Override the suite's fail_below_iqs threshold."
        ),
        json_output: bool = typer.Option(
            False, "--json", help="Print a machine-readable JSON summary instead of text."
        ),
        output: str = typer.Option(
            None, "--output", help="Write a JUnit XML report to this path (for CI)."
        ),
    ):
        """Run a YAML-defined quality regression suite (CI/CD quality gate)."""
        import json as json_module
        from scroot.cli.eval import format_junit_xml, format_report, load_suite, run_suite

        try:
            suite_obj = load_suite(suite)
        except Exception as e:
            typer.echo(f"ERROR: {e}")
            raise typer.Exit(1)

        result = run_suite(suite_obj, fail_below=fail_below)

        if output:
            with open(output, "w", encoding="utf-8") as f:
                f.write(format_junit_xml(suite_obj, result))

        if json_output:
            typer.echo(json_module.dumps({
                "name": suite_obj.name,
                "passed": result.passed_count,
                "failed": result.failed_count,
                "avg_iqs": result.avg_iqs,
                "results": [
                    {
                        "query": r.example.query,
                        "iqs": r.iqs,
                        "passed": r.passed,
                        "gate_reason": r.gate_reason,
                        "tags": r.example.tags,
                    }
                    for r in result.results
                ],
            }, indent=2))
        else:
            typer.echo(format_report(suite_obj, result))

        if result.failed_count:
            raise typer.Exit(1)

    @app.command()
    def serve(
        port:  int = typer.Option(7432,          help="Port to listen on"),
        store: str = typer.Option("./scroot_store.jsonl", help="JSONL feedback store path"),
        host:  str = typer.Option("127.0.0.1",   help="Host to bind to"),
        token: str = typer.Option(
            None, "--token",
            help="Require this shared token on all /api routes (for network "
                 "binds). Falls back to SCROOT_DASHBOARD_TOKEN.",
        ),
        hosted: bool = typer.Option(False,        hidden=True),
    ):
        """Start the Scroot Review Console at http://localhost:7432

        The dashboard has no per-user login. The default 127.0.0.1 bind is
        single-user safe. If you bind to a routable host (e.g. --host 0.0.0.0),
        set --token (or SCROOT_DASHBOARD_TOKEN) and/or front it with an
        authenticating reverse proxy - otherwise the correction store and the
        stored LLM API key are reachable by anyone on the network.
        """
        try:
            import uvicorn
        except ImportError:
            typer.echo("ERROR: Install dashboard deps: pip install 'scroot[dashboard]'")
            raise typer.Exit(1)

        from scroot.dashboard.security import is_loopback_host, resolve_dashboard_token
        from scroot.dashboard.server import create_app

        fa_app = create_app(store_path=store, hosted=hosted, host=host, auth_token=token)

        typer.echo("\n  * SCROOT Review Console")
        typer.echo(f"  Store : {store}")
        typer.echo(f"  URL   : http://{host}:{port}")
        if resolve_dashboard_token(token) is not None:
            typer.echo("  Auth  : token required (Authorization: Bearer / X-Scroot-Token)")
        elif not is_loopback_host(host):
            typer.echo(
                "  WARNING: non-loopback bind with NO auth - the store and "
                "stored API key are exposed to the network. Set --token."
            )
        typer.echo("")
        uvicorn.run(fa_app, host=host, port=port, log_level="info")

except ImportError:
    # typer not installed - provide a minimal fallback
    import sys

    class _FakeCLI:
        def command(self, *a, **kw):
            def dec(fn): return fn
            return dec
        def __call__(self):
            print("scroot: install typer for CLI support: pip install typer")
            sys.exit(1)

    app = _FakeCLI()
