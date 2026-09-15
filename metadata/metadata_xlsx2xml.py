#  Copyright notice
#   --------------------------------------------------------------------
#   Copyright (C) 2024 Deltares Stars4Water
#   Gerrit.Hendriksen@deltares.nl
#
#   This library is free software: you can redistribute it and/or modify
#   it under the terms of the GNU General Public License as published by
#   the Free Software Foundation, either version 3 of the License, or
#   (at your option) any later version.
#
#   This library is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#   GNU General Public License for more details.
#
#   You should have received a copy of the GNU General Public License
#   along with this library.  If not, see <http://www.gnu.org/licenses/>.
#   --------------------------------------------------------------------
#
# This tool is part of <a href="http://www.OpenEarth.eu">OpenEarthTools</a>.
# OpenEarthTools is an online collaboration to share and manage data and
# programming tools in an open source, version controlled environment.
# Sign up to recieve regular updates of this function, and to contribute
# your own tools.

"""
Convert metadata stored in Excel to xmls
"""
import os
import numpy as np
from datetime import time
import pandas as pd
import uuid
from xml.sax.saxutils import escape

"""
directory where metadata is stored
"""
tdir = r"C:\projectinfo\eu\metadata"

"""setup mapping between metadata entries and xlsx"""
maptable = r".\mappingtable.xlsx"
d = (
    pd.read_excel(maptable, sheet_name="Sheet1", index_col=0, header=None)
    .transpose()
    .to_dict("records")[0]
)

"""
read the xslx with the metadata
Check column headers first, in some cases there are extra spaces there!!!
"""

mxlsx = r"C:\projectinfo\eu\stars4water\work\Additional_Dataset_for_portal_Sept2025_v2.xlsx"
mxlsx = r"C:\projectinfo\eu\stars4water\work\Local datasets version 2026-03-02.xlsx"
sn = "S4W Datasets"
mxlsx = r"C:\projectinfo\eu\stars4water\work\Datasets_for_portal_20260825.xlsx"
sn = 'Sheet1'
mxlsx = r"C:\projectinfo\eu\stars4water\work\Additional_Dataset_for_portal_Sept2025_v2.xlsx"
sn = "S4W Datasets"

dctxls = {}
dctxls['a'] = ("Additional_Dataset_for_portal_Sept2025.xlsx","S4W Datasets")
dctxls['b'] = ("Additional_Dataset_for_portal_Sept2025_v2.xlsx","S4W Datasets")
dctxls['c'] = ("Datasets_for_portal_20260825.xlsx","Sheet1")
dctxls['d'] = ("Datasets_for_portal20240324.xlsx","Sheet1")
dctxls['e'] = ("Datasets_for_portal20240726.xlsx","Sheet1")
dctxls['f'] = ("Local datasets version 2026-03-02.xlsx","S4W Datasets")


"""
Read the xmltemplate
"""
txml = r".\iso19139_template.xml"
pathnm = r"C:\projectinfo\eu\stars4water\work"

for key, (file, sheet) in dctxls.items():
    print(f"Key: {key}, File: {file}, Sheet: {sheet}")

    dfm = pd.read_excel(
        os.path.join(pathnm, file), skiprows=2, sheet_name=sheet, na_filter=False, index_col=0
    )

    """
    Some sheets use "Roll" instead of "Role" as column header, normalize it
    """
    if "Roll" in dfm.columns and "Role" not in dfm.columns:
        dfm = dfm.rename(columns={"Roll": "Role"})
    if "Roll" in d and "Role" not in d:
        d["Role"] = d.pop("Roll")

    for r in range(len(dfm)):
        title = dfm["Title of the source"].iloc[r]
        if title != "":
            "create new file"
            fn = os.path.join(tdir, ".".join([title.replace(" ", ""), "xml"]))

            "open the template"
            fxml = open(txml, "r+")
            xml = fxml.read()
            fxml.close()

            "define unique identifier"
            uid = str(uuid.uuid4())
            

            xml = xml.replace("{uuid}", uid)
            for k in d.keys():
                try:
                    value = escape(str(dfm[k].iloc[r]))
                    xml = xml.replace("{" + d[k] + "}", value)
                except Exception as e:
                    value = repr(dfm[k].iloc[r]) if k in dfm.columns else "<missing column>"
                    print("it went wrong with", k, d[k], value, "-", e)
            with open(fn, "w+", encoding="utf-8") as fm:
                fm.write(xml)
            print(uid, fn)