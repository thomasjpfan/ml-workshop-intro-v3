preprocessor = ColumnTransformer(
    [
        (
            "categorical",
            TargetEncoder(),
            make_column_selector(dtype_include="category"),
        ),
        ("numerical", SimpleImputer(), make_column_selector(dtype_include="number")),
    ],
    verbose_feature_names_out=False,
)

pipe = make_pipeline(preprocessor, RandomForestClassifier(random_state=0))

pipe.fit(X_train, y_train)

pipe.score(X_test, y_test)

rf = pipe[-1]

rf.feature_names_in_

rf.feature_importances_

importances_series = pd.Series(
    rf.feature_importances_, index=rf.feature_names_in_
).sort_values()

importances_series.plot(kind="barh")
