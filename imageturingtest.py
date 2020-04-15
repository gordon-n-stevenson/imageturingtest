#!/usr/bin/env python
# -*- coding: utf-8 -*-
# * =========================================================================
# *
# *  Copyright Gordon Stevenson. Perinatal Ultrasound Imaging Group, UNSW Sydney.
# *
# *  Licensed under the Apache License, Version 2.0 (the "License");
# *  you may not use this file except in compliance with the License.
# *  You may obtain a copy of the License at
# *
# *         http://www.apache.org/licenses/LICENSE-2.0.txt
# *
# *  Unless required by applicable law or agreed to in writing, software
# *  distributed under the License is distributed on an "AS IS" BASIS,
# *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# *  See the License for the specific language governing permissions and
# *  limitations under the License.
# * =========================================================================

"""Run Image Turing Test.

Usage:
    imageturingtest.py <gtdir> <preddir>
    imageturingtest.py <gtdir> <preddir> [<output_name>]
    imageturingtest.py [-hv]

Options:
    -h --help            show this message and exit.
    --version            show version information and exit.
"""

from __future__ import print_function
from docopt import docopt
import SimpleITK as sitk
import vtk
from numpy import *
import glob
import pandas as pd

__version__ = '0.1'

# Adapted from the VTK example
# http://www.vtk.org/Wiki/VTK/Examples/Python/vtkWithNumpy
# dictionary to convert SimpleITK pixel types to VTK
pixelmap = {sitk.sitkUInt8:   vtk.VTK_UNSIGNED_CHAR,  sitk.sitkInt8:    vtk.VTK_CHAR,
            sitk.sitkUInt16:  vtk.VTK_UNSIGNED_SHORT, sitk.sitkInt16:   vtk.VTK_SHORT,
            sitk.sitkUInt32:  vtk.VTK_UNSIGNED_INT,   sitk.sitkInt32:   vtk.VTK_INT,
            sitk.sitkUInt64:  vtk.VTK_UNSIGNED_LONG,  sitk.sitkInt64:   vtk.VTK_LONG,
            sitk.sitkFloat32: vtk.VTK_FLOAT,          sitk.sitkFloat64: vtk.VTK_DOUBLE,

            sitk.sitkVectorUInt8:   vtk.VTK_UNSIGNED_CHAR,  sitk.sitkVectorInt8:    vtk.VTK_CHAR,
            sitk.sitkVectorUInt16:  vtk.VTK_UNSIGNED_SHORT, sitk.sitkVectorInt16:   vtk.VTK_SHORT,
            sitk.sitkVectorUInt32:  vtk.VTK_UNSIGNED_INT,   sitk.sitkVectorInt32:   vtk.VTK_INT,
            sitk.sitkVectorUInt64:  vtk.VTK_UNSIGNED_LONG,  sitk.sitkVectorInt64:   vtk.VTK_LONG,
            sitk.sitkVectorFloat32: vtk.VTK_FLOAT,          sitk.sitkVectorFloat64: vtk.VTK_DOUBLE,

            sitk.sitkLabelUInt8:  vtk.VTK_UNSIGNED_CHAR,
            sitk.sitkLabelUInt16: vtk.VTK_UNSIGNED_SHORT,
            sitk.sitkLabelUInt32: vtk.VTK_UNSIGNED_INT,
            sitk.sitkLabelUInt64: vtk.VTK_UNSIGNED_LONG}

def sitk2vtk(img, outVol=None, debugOn=False):
    """
    converts a sitk image, img and returns it to vtkImageData
    """
    """Convert a SimpleITK image to a VTK image, via numpy."""

    size = list(img.GetSize())
    origin = list(img.GetOrigin())
    spacing = list(img.GetSpacing())
    sitktype = img.GetPixelID()
    vtktype = pixelmap[sitktype]
    ncomp = img.GetNumberOfComponentsPerPixel()

    # convert the SimpleITK image to a numpy array
    i2 = sitk.GetArrayFromImage(img)
    i2_string = i2.tostring()
    if debugOn:
        print("data string address inside sitk2vtk", hex(id(i2_string)))

    # send the numpy array to VTK with a vtkImageImport object
    dataImporter = vtk.vtkImageImport()
    dataImporter.CopyImportVoidPointer(i2_string, len(i2_string))
    dataImporter.SetDataScalarType(vtktype)
    dataImporter.SetNumberOfScalarComponents(ncomp)

    # VTK expects 3-dimensional parameters
    if len(size) == 2:
        size.append(1)

    if len(origin) == 2:
        origin.append(0.0)

    if len(spacing) == 2:
        spacing.append(spacing[0])

    # Set the new VTK image's parameters
    dataImporter.SetDataExtent(0, size[0]-1, 0, size[1]-1, 0, size[2]-1)
    dataImporter.SetWholeExtent(0, size[0]-1, 0, size[1]-1, 0, size[2]-1)
    dataImporter.SetDataOrigin(origin)
    dataImporter.SetDataSpacing(spacing)
    dataImporter.Update()

    vtk_image = dataImporter.GetOutput()
    outVol = vtk.vtkImageData()
    outVol.DeepCopy(vtk_image)

    if debugOn:
        print("Volume object inside sitk2vtk")
        print(vtk_image)
        print("type = ", vtktype)
        print("num components = ", ncomp)
        print(size)
        print(origin)
        print(spacing)
        print(vtk_image.GetScalarComponentAsFloat(0, 0, 0, 0))

    return outVol, size, origin, spacing

def smooth_and_contour(img, seg_id):

    img = sitk.ReadImage(img)
    img = sitk.BinaryThreshold(img, lowerThreshold=seg_id, upperThreshold=seg_id, insideValue=1, outsideValue=0)
    gaussian = sitk.SmoothingRecursiveGaussianImageFilter()
    gaussian.SetSigma(1)
    img = gaussian.Execute(img)
    bin_img = sitk.BinaryThreshold(img, lowerThreshold=1-0.5, upperThreshold=1+0.5, insideValue=1, outsideValue=0)
    bin_img = sitk.BinaryContour(bin_img, fullyConnected=False)
    return bin_img

def clearRenderer(rend):
    #given a renderer, runs over the actors and removes it all.
    for i in rend.GetActors():
        rend.RemoveActor(i)
    rend.RemoveAllViewProps()
    return

def load_images(pat_id, gt_dir_path, pred_path, data_path=None):

    seg_path = glob.glob('{}/*_{}_*.nii.gz'.format(gt_dir_path, pat_id, pat_id))[0]
    pred_path = glob.glob('{}/*{}.nii.gz'.format(pred_path, pat_id))[0]

    if data_path is None:
        data_path  = 'E:/PlacentaData/HPP_TestData'
    bmode_path = glob.glob('{}/{}/*_{}.*'.format(data_path, pat_id, pat_id))[0]

    bmode_path.replace('\\', '/')
    pred_path.replace('\\', '/')
    seg_path.replace('\\', '/')

    return bmode_path, seg_path, pred_path

def getLuT():
    #builds four lookup tables for the b-mode ultrasound and cmaps for the binary segmentations
    bmode_lut = vtk.vtkLookupTable()
    bmode_lut.SetNumberOfColors(256)
    for x in range(0, 256):
        bmode_lut.SetTableValue(x, x / 256, x / 256, x / 256, 0.95)
    bmode_lut.SetTableRange(0, 256)
    bmode_lut.Build()

    fetus_lut = vtk.vtkLookupTable()
    fetus_lut.SetNumberOfColors(2)
    fetus_lut.SetTableValue(0, 0.0, 0.0, 0.0, 0.0)
    fetus_lut.SetTableValue(1, 230 / 256, 75 / 256, 53 / 256, 1.0)
    fetus_lut.SetTableRange(0, 1)
    fetus_lut.Build()

    plac_lut = vtk.vtkLookupTable()
    plac_lut.SetNumberOfColors(2)
    plac_lut.SetTableValue(0, 0.0, 0.0, 0.0, 0.0)
    plac_lut.SetTableValue(1, 77 / 255, 187 / 255, 213 / 255, 255 / 255)
    plac_lut.SetTableRange(0, 1)
    plac_lut.Build()

    amnio_lut = vtk.vtkLookupTable()
    amnio_lut.SetNumberOfColors(2)
    amnio_lut.SetTableValue(0, 0.0, 0.0, 0.0, 0.0)
    amnio_lut.SetTableValue(1, 0 / 255, 160 / 255, 135 / 255, 250 / 255)
    amnio_lut.SetTableRange(0, 1)
    amnio_lut.Build()

    return bmode_lut, fetus_lut, plac_lut, amnio_lut

def main():

    arguments = docopt(__doc__, version = '0.1')

    global next, toss, exp_method, gt_dir_path, pred_path

    pred_path = arguments['<preddir>']
    gt_dir_path = arguments['<gtdir>']

    pred_path.replace('\\', '/')
    gt_dir_path.replace('\\', '/')

    if arguments['<output_name>'] is not None:
        output_path = arguments['<output_name>']
    else:
        output_path = 'result.csv'

    #pat_id - the pred was either 0 or 1 (left/right); my guess using arrow key (left/right)
    results = []
    next = 0

    #so if we do this multiple times we don't see the images in sequence over and over.
    pat_list = glob.glob('{}/*.nii.gz'.format(pred_path))
    random.shuffle(pat_list)

    pat_ids = [i.split('_')[-1].split('.')[0] for i in pat_list]

    img_bmode_path, img_seg_path, img_pred_path = load_images(pat_ids[next], gt_dir_path, pred_path)
    arr_bmode = sitk.GetArrayFromImage(sitk.ReadImage(img_bmode_path))
    bmode = sitk.ReadImage(img_bmode_path)

    img_bmode_vtk = sitk.GetImageFromArray(arr_bmode)
    img_bmode_vtk.SetOrigin(bmode.GetOrigin())
    img_bmode_vtk.SetSpacing(bmode.GetSpacing())

    img_placenta = smooth_and_contour(img_seg_path, 1)
    img_amnio = smooth_and_contour(img_seg_path, 2)
    img_fetus = smooth_and_contour(img_seg_path, 3)

    vol_placenta, sizef, originf, spacingf = sitk2vtk(img_placenta, debugOn=False)
    vol_fetus, _, _, _ = sitk2vtk(img_fetus, debugOn=False)
    vol_amnio, _, _, _ = sitk2vtk(img_amnio, debugOn=False)

    img_placenta_pred = smooth_and_contour(img_pred_path, 1)
    img_amnio_pred = smooth_and_contour(img_pred_path, 2)
    img_fetus_pred = smooth_and_contour(img_pred_path, 3)

    vol_placenta_pred, sizef, originf, spacingf = sitk2vtk(img_placenta_pred, debugOn=False)
    vol_fetus_pred, _, _, _ = sitk2vtk(img_fetus_pred, debugOn=False)
    vol_amnio_pred, _, _, _ = sitk2vtk(img_amnio_pred, debugOn=False)

    vol, size, origin, spacing = sitk2vtk(img_bmode_vtk, debugOn=False)

    # Calculate the center of the volume
    (xMin, xMax, yMin, yMax, zMin, zMax) = vol.GetExtent()
    (xSpacing, ySpacing, zSpacing) = vol.GetSpacing()
    (x0, y0, z0) = vol.GetOrigin()

    center = [x0 + xSpacing * 0.5 * (xMin + xMax),
              y0 + ySpacing * 0.5 * (yMin + yMax),
              z0 + zSpacing * 0.5 * (zMin + zMax)]

    colors_vtk = vtk.vtkNamedColors()
    color_lbls = ["Col1", "Col2", "Col3", "Col4", "BkGrd"]
    colors_vtk.SetColor("Col1", [230, 75, 53, 255])
    colors_vtk.SetColor("Col2", [77, 187, 213, 255])
    colors_vtk.SetColor("Col3", [0, 160, 135, 255])
    colors_vtk.SetColor("Col4", [60, 84, 136, 255])
    colors_vtk.SetColor("BkGrd", [225, 225, 225, 255])

    # Matrices for orientations
    sagittal = vtk.vtkMatrix4x4()
    sagittal.DeepCopy((0, 0, -1, center[0],
                       1, 0, 0, center[1],
                       0, 1, 0, center[2],
                       0, 0, 0, 1))

    # Extract a slice in the desired orientation
    reslice = vtk.vtkImageReslice()
    reslice.SetInputData(vol)
    reslice.SetOutputDimensionality(2)
    reslice.SetResliceAxes(sagittal)
    reslice.SetInterpolationModeToLinear()

    # Extract a slice in the desired orientation
    reslice_fetus = vtk.vtkImageReslice()
    reslice_fetus.SetInputData(vol_fetus)
    reslice_fetus.SetOutputDimensionality(2)
    reslice_fetus.SetResliceAxes(sagittal)
    reslice_fetus.SetInterpolationModeToLinear()

    # Extract a slice in the desired orientation
    reslice_placenta = vtk.vtkImageReslice()
    reslice_placenta.SetInputData(vol_placenta)
    reslice_placenta.SetOutputDimensionality(2)
    reslice_placenta.SetResliceAxes(sagittal)
    reslice_placenta.SetInterpolationModeToLinear()

    # Extract a slice in the desired orientation
    reslice_amnio = vtk.vtkImageReslice()
    reslice_amnio.SetInputData(vol_amnio)
    reslice_amnio.SetOutputDimensionality(2)
    reslice_amnio.SetResliceAxes(sagittal)
    reslice_amnio.SetInterpolationModeToLinear()

    # Extract a slice in the desired orientation
    reslice_fetus_pred = vtk.vtkImageReslice()
    reslice_fetus_pred.SetInputData(vol_fetus_pred)
    reslice_fetus_pred.SetOutputDimensionality(2)
    reslice_fetus_pred.SetResliceAxes(sagittal)
    reslice_fetus_pred.SetInterpolationModeToLinear()

    # Extract a slice in the desired orientation
    reslice_placenta_pred = vtk.vtkImageReslice()
    reslice_placenta_pred.SetInputData(vol_placenta_pred)
    reslice_placenta_pred.SetOutputDimensionality(2)
    reslice_placenta_pred.SetResliceAxes(sagittal)
    reslice_placenta_pred.SetInterpolationModeToLinear()

    # Extract a slice in the desired orientation
    reslice_amnio_pred = vtk.vtkImageReslice()
    reslice_amnio_pred.SetInputData(vol_amnio_pred)
    reslice_amnio_pred.SetOutputDimensionality(2)
    reslice_amnio_pred.SetResliceAxes(sagittal)
    reslice_amnio_pred.SetInterpolationModeToLinear()

    bmode_lut, fetus_lut, plac_lut, amnio_lut = getLuT()

    # Map the image through the lookup table
    color_fetus = vtk.vtkImageMapToColors()
    color_fetus.SetLookupTable(fetus_lut)
    color_fetus.SetInputConnection(reslice_fetus.GetOutputPort())

    # Map the image through the lookup table
    color_plac = vtk.vtkImageMapToColors()
    color_plac.SetLookupTable(plac_lut)
    color_plac.SetInputConnection(reslice_placenta.GetOutputPort())

    # Map the image through the lookup table
    color_amnio = vtk.vtkImageMapToColors()
    color_amnio.SetLookupTable(amnio_lut)
    color_amnio.SetInputConnection(reslice_amnio.GetOutputPort())

    # Map the image through the lookup table
    color_fetus_pred = vtk.vtkImageMapToColors()
    color_fetus_pred.SetLookupTable(fetus_lut)
    color_fetus_pred.SetInputConnection(reslice_fetus_pred.GetOutputPort())

    # Map the image through the lookup table
    color_plac_pred = vtk.vtkImageMapToColors()
    color_plac_pred.SetLookupTable(plac_lut)
    color_plac_pred.SetInputConnection(reslice_placenta_pred.GetOutputPort())

    # Map the image through the lookup table
    color_amnio_pred = vtk.vtkImageMapToColors()
    color_amnio_pred.SetLookupTable(amnio_lut)
    color_amnio_pred.SetInputConnection(reslice_amnio_pred.GetOutputPort())

    # Map the image through the lookup table
    color = vtk.vtkImageMapToColors()
    color.SetLookupTable(bmode_lut)
    color.SetInputConnection(reslice.GetOutputPort())

    # Display the image
    actor = vtk.vtkImageActor()
    actor.GetMapper().SetInputConnection(color.GetOutputPort())
    actor.RotateX(180)

    actor_fetus = vtk.vtkImageActor()
    actor_fetus.GetMapper().SetInputConnection(color_fetus.GetOutputPort())
    actor_fetus.RotateX(180)

    actor_plac = vtk.vtkImageActor()
    actor_plac.GetMapper().SetInputConnection(color_plac.GetOutputPort())
    actor_plac.RotateX(180)

    actor_amnio = vtk.vtkImageActor()
    actor_amnio.GetMapper().SetInputConnection(color_amnio.GetOutputPort())
    actor_amnio.RotateX(180)

    actor_fetus_pred = vtk.vtkImageActor()
    actor_fetus_pred.GetMapper().SetInputConnection(color_fetus_pred.GetOutputPort())
    actor_fetus_pred.RotateX(180)

    actor_plac_pred = vtk.vtkImageActor()
    actor_plac_pred.GetMapper().SetInputConnection(color_plac_pred.GetOutputPort())
    actor_plac_pred.RotateX(180)

    actor_amnio_pred = vtk.vtkImageActor()
    actor_amnio_pred.GetMapper().SetInputConnection(color_amnio_pred.GetOutputPort())
    actor_amnio_pred.RotateX(180)

    renderer_a = vtk.vtkRenderer()
    renderer_b = vtk.vtkRenderer()

    renderer_a.AddActor(actor)
    renderer_a.AddActor(actor_fetus)
    renderer_a.AddActor(actor_amnio)
    renderer_a.AddActor(actor_plac)

    renderer_b.AddActor(actor)
    renderer_b.AddActor(actor_fetus_pred)
    renderer_b.AddActor(actor_amnio_pred)
    renderer_b.AddActor(actor_plac_pred)

    # Set up the interaction
    interactorStyle = vtk.vtkInteractorStyleImage()
    interactor = vtk.vtkRenderWindowInteractor()
    interactor.SetInteractorStyle(interactorStyle)

    window = vtk.vtkRenderWindow()
    window.SetInteractor(interactor)
    window.SetSize(1000, 400)
    window.AddRenderer(renderer_a)
    window.AddRenderer(renderer_b)

    textActor = vtk.vtkTextActor()
    textActor.SetInput("Pat Id = {} \t {} of {}".format(pat_ids[next], next + 1, len(pat_ids)))
    textActor.SetPosition2(20, 40)
    textActor.GetTextProperty().SetFontSize(24)
    textActor.GetTextProperty().SetColor(0, 0, 0)

    global toss
    toss = random.rand()
    #puts the human on the left; AI on the right
    if toss >= 0.5:
        renderer_a.SetViewport(0, 0, 0.5, 1)
        renderer_b.SetViewport(0.5,0, 1.0, 1)
        renderer_a.AddActor(textActor)
    else:
        # puts the human on the right; AI on the right
        renderer_b.SetViewport(0, 0, 0.5, 1)
        renderer_a.SetViewport(0.5, 0, 1.0, 1)
        renderer_b.AddActor(textActor)

    renderer_a.SetBackground(0.9,0.9,0.9)
    renderer_b.SetBackground(0.9,0.9,0.9)

    renderer_a.ResetCamera()
    renderer_b.ResetCamera()

    renderer_a.GetActiveCamera().Zoom(1.4)
    renderer_b.GetActiveCamera().Zoom(1.4)

    window.Render()

    # Create callbacks for slicing the image
    actions = {}
    actions["Slicing"] = 0

    def ButtonCallback(obj, event):
        if event == "LeftButtonPressEvent":
            actions["Slicing"] = 1
        else:
            actions["Slicing"] = 0

    def MouseMoveCallback(obj, event):
        (lastX, lastY) = interactor.GetLastEventPosition()
        (mouseX, mouseY) = interactor.GetEventPosition()
        if actions["Slicing"] == 1:
            deltaY = mouseY - lastY

            sliceSpacing = reslice.GetOutput().GetSpacing()[2]
            for matrix in [reslice.GetResliceAxes(), reslice_fetus_pred.GetResliceAxes()]:
                # move the center point that we are slicing through
                center = matrix.MultiplyPoint((0, 0, sliceSpacing*deltaY, 1))
                matrix.SetElement(0, 3, center[0])
                matrix.SetElement(1, 3, center[1])
                matrix.SetElement(2, 3, center[2])


            window.Render()
        else:
            interactorStyle.OnMouseMove()


    def onMouseWheelBackwardEvent(sender, event):
        pass

    def onMouseWheelForwardEvent(sender, event):
        pass

    def KeyboardCallback(obj, event):
        key = interactor.GetKeySym()
        #
        if ('Left' == interactor.GetKeySym()) or ('Right' == interactor.GetKeySym()):

            if globals()['next'] == len(pat_ids)-1:
                textActor.SetInput('Last Patient!')
                window.Render()

                df = pd.DataFrame(results)
                df.columns = ['PatId', 'HumanWasWhere', 'GuessedHere']
                df.to_csv()
                print('Save to {} \t'.format())
                return

            if globals()['toss'] >= 0.5:
                human = 0
            else:
                human = 1

            if interactor.GetKeySym() == 'Left':
                results.append([pat_ids[next], human, 0])
            else:
                results.append([pat_ids[next], human, 1])

            globals()['next'] += 1
            img_bmode_path, img_seg_path, img_pred_path = load_images(pat_ids[next], gt_dir_path, pred_path)

            arr_bmode = sitk.GetArrayFromImage(sitk.ReadImage(img_bmode_path))
            bmode = sitk.ReadImage(img_bmode_path)

            img_bmode_vtk = sitk.GetImageFromArray(arr_bmode)
            img_bmode_vtk.SetOrigin(bmode.GetOrigin())
            img_bmode_vtk.SetSpacing(bmode.GetSpacing())

            img_placenta = smooth_and_contour(img_seg_path, 1)
            img_amnio = smooth_and_contour(img_seg_path, 2)
            img_fetus = smooth_and_contour(img_seg_path, 3)

            vol_placenta, sizef, originf, spacingf = sitk2vtk(img_placenta, debugOn=False)
            vol_fetus, _, _, _ = sitk2vtk(img_fetus, debugOn=False)
            vol_amnio, _, _, _ = sitk2vtk(img_amnio, debugOn=False)

            img_placenta_pred = smooth_and_contour(img_pred_path, 1)
            img_amnio_pred = smooth_and_contour(img_pred_path, 2)
            img_fetus_pred = smooth_and_contour(img_pred_path, 3)

            vol_placenta_pred, sizef, originf, spacingf = sitk2vtk(img_placenta_pred, debugOn=False)
            vol_fetus_pred, _, _, _ = sitk2vtk(img_fetus_pred, debugOn=False)
            vol_amnio_pred, _, _, _ = sitk2vtk(img_amnio_pred, debugOn=False)

            vol, size, origin, spacing = sitk2vtk(img_bmode_vtk, debugOn=False)

            if globals()['toss'] >= 0.5:
                renderer_a.RemoveActor(textActor)
            else:
                renderer_b.RemoveActor(textActor)

            reslice.SetInputData(vol)
            reslice_fetus.SetInputData(vol_fetus)
            reslice_fetus_pred.SetInputData(vol_fetus_pred)

            reslice_placenta.SetInputData(vol_placenta)
            reslice_placenta_pred.SetInputData(vol_placenta_pred)

            reslice_amnio.SetInputData(vol_amnio)
            reslice_amnio_pred.SetInputData(vol_amnio_pred)

            window.Render()

            globals()['toss'] =  random.rand()
            textActor.SetInput("Pat Id = {} \t {} of {}".format(pat_ids[next], next+1, len(pat_ids)))
            if toss >= 0.5:
                renderer_a.SetViewport(0, 0, 0.5, 1)
                renderer_b.SetViewport(0.5, 0, 1.0, 1)
                renderer_a.AddActor(textActor)
            else:
                renderer_b.SetViewport(0, 0, 0.5, 1)
                renderer_a.SetViewport(0.5, 0, 1.0, 1)
                renderer_b.AddActor(textActor)

            window.Render()

            renderer_a.ResetCamera()
            renderer_b.ResetCamera()

            renderer_a.GetActiveCamera().Zoom(1.4)
            renderer_b.GetActiveCamera().Zoom(1.4)

            pass
        if 'q' == interactor.GetKeySym():
            pass
        #key handling

    interactorStyle.AddObserver("MouseMoveEvent", MouseMoveCallback)
    interactorStyle.AddObserver("LeftButtonPressEvent", ButtonCallback)
    interactorStyle.AddObserver("LeftButtonReleaseEvent", ButtonCallback)
    interactorStyle.AddObserver("KeyPressEvent", KeyboardCallback, 1)
    interactorStyle.AddObserver("MouseWheelForwardEvent", onMouseWheelForwardEvent)
    # Start interaction
    interactor.Start()

    del renderer_a
    del renderer_b
    del window
    del interactor

if __name__ == '__main__':
    main()