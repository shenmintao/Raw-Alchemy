import click
from raw_alchemy import lensfun_wrapper as lf
from raw_alchemy import config, orchestrator

@click.command()
@click.argument("input_path", type=click.Path(exists=True))
@click.argument("output_path", type=click.Path())
@click.option(
    "--log-space",
    required=True,
    type=click.Choice(list(config.LOG_TO_WORKING_SPACE.keys()), case_sensitive=False),
    help="The log space to convert to.",
)
@click.option(
    "--lut",
    "lut_path",
    type=click.Path(exists=True),
    help="Path to a .cube LUT file to apply.",
)
@click.option(
    "--exposure",
    type=float,
    default=None,
    help="Manual exposure adjustment in stops (e.g., -0.5, 1.0). Overrides all auto exposure.",
)
@click.option(
    "--lens-correct",
    default=True,
    help="Enable or disable lens distortion correction. Enabled by default.",
)
@click.option(
    "--custom-lensfun-db",
    "custom_lensfun_db_path",
    type=click.Path(exists=True),
    help="Path to a custom lensfun database XML file.",
)
@click.option(
    "--metering",
    default="hybrid",
    type=click.Choice(config.METERING_MODES, case_sensitive=False),
    help="Auto exposure metering mode: hybrid (default), average, center-weighted, highlight-safe.",
)
@click.option(
    "--jobs",
    type=int,
    default=4,
    help="Number of concurrent jobs for batch processing. Default is 4.",
)
@click.option(
    "--format",
    "output_format",
    type=click.Choice(['tif', 'heif', 'hdr-heif', 'jpg'], case_sensitive=False),
    default='tif',
    help="Output file format. Use hdr-heif for BT.2020/PQ HEIF. Default is 'tif'.",
)
def main(input_path, output_path, log_space, lut_path, exposure, lens_correct, custom_lensfun_db_path, metering, jobs, output_format):
    """
    Converts RAW image(s) to high-quality image files (TIFF, HEIF, or JPG).

    INPUT_PATH: Path to a single RAW file or a directory of RAWs.
    OUTPUT_PATH: Path to the output file or a directory for batch processing.
    """
    try:
        orchestrator.process_path(
            input_path=input_path,
            output_path=output_path,
            log_space=log_space,
            lut_path=lut_path,
            exposure=exposure,
            lens_correct=lens_correct,
            custom_db_path=custom_lensfun_db_path,
            metering_mode=metering,
            jobs=jobs,
            logger_func=click.echo, # Use click.echo for robust Unicode support
            output_format=output_format,
        )
    except Exception as e:
        # The orchestrator will log specifics, but we can catch fatal errors here.
        raise click.ClickException(f"A critical error occurred: {e}")


if __name__ == "__main__":
    main()
