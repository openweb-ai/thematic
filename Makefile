.FORCE:

requirements:
	poetry export -f requirements.txt --output requirements.txt --extras all
	poetry export -f requirements.txt --output requirements-dev.txt --dev --extras all

docs: .FORCE requirements
	poetry run mkdocs build

check:
	poetry run isort -c openweb
	poetry run black --check openweb

check-codes:
	set -e
	poetry run flake8
	poetry run pylint openweb
	poetry run mypy openweb
	poetry run bandit -r openweb/ -x *_test.py

safety:
	poetry run safety check

test: check check-codes
	set -e
	poetry run pytest openweb --cov

coveralls: test
	poetry run coveralls

serve-docs: docs
	cd site/  && poetry run python -m http.server 8000

format:
	poetry run autoflake -i -r ./openweb --remove-all-unused-imports --ignore-init-module-imports --expand-star-imports
	poetry run isort openweb
	poetry run black openweb
